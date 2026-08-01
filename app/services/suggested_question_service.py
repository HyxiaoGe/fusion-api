"""推荐问题生成、持久化与并发时序控制。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import litellm
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.ai.llm_observability import merge_litellm_kwargs
from app.ai.prompts import prompt_manager
from app.core.logger import app_logger as logger
from app.db.models import Conversation as ConversationModel
from app.db.models import Message as MessageModel
from app.services.chat.utils import ChatUtils

SUGGESTION_STATUS_IDLE = "idle"
SUGGESTION_STATUS_PENDING = "pending"
SUGGESTION_STATUS_READY = "ready"
SUGGESTION_STATUS_FAILED = "failed"


class SuggestedQuestionTargetNotFoundError(ValueError):
    """目标消息不存在、越权或不是 assistant；对外统一按不存在处理。"""


class SuggestedQuestionTargetInvalidError(ValueError):
    """目标消息存在但没有可用于生成推荐问题的正式正文。"""


@dataclass(frozen=True)
class SuggestedQuestionClaim:
    message_id: str
    conversation_id: str
    revision: int
    model_id: str


@dataclass(frozen=True)
class SuggestedQuestionRequestClaim:
    claim: SuggestedQuestionClaim | None
    questions: list[str]
    message_id: str
    revision: int
    status: str


@dataclass(frozen=True)
class SuggestedQuestionGenerationResult:
    questions: list[str]
    message_id: str
    revision: int
    status: str
    applied: bool


class SuggestedQuestionService:
    """以 assistant message 为唯一目标管理推荐问题。"""

    UTILITY_MODEL_ID = "deepseek-chat"
    UTILITY_LLM_TIMEOUT = 8
    MAX_TOKENS = 512
    FALLBACK_QUESTIONS = [
        "您对这个主题还有其他问题吗？",
        "您想了解更多相关信息吗？",
        "您想要探讨这个话题的哪些方面？",
    ]

    def __init__(self, db: Session):
        self.db = db

    def claim_auto_generation(
        self,
        *,
        assistant_message_id: str,
        run_id: str,
    ) -> SuggestedQuestionClaim | None:
        """为成功终态领取一次自动生成；同一 run 幂等，新 continuation 可刷新。"""

        try:
            message = self._locked_message(assistant_message_id=assistant_message_id)
            if message is None or message.role != "assistant" or not self._formal_text(message.content):
                self.db.rollback()
                return None
            if message.suggested_questions_auto_run_id == run_id:
                self.db.rollback()
                return None

            claim = self._advance_revision(message)
            message.suggested_questions_auto_run_id = run_id
            self.db.commit()
            return claim
        except Exception:
            self.db.rollback()
            raise

    def claim_request_generation(
        self,
        *,
        conversation_id: str,
        user_id: str,
        assistant_message_id: str | None,
        force_refresh: bool,
    ) -> SuggestedQuestionRequestClaim:
        """领取 API 请求版本；force refresh 总是抢占更高 revision。"""

        try:
            message = self._locked_message(
                conversation_id=conversation_id,
                user_id=user_id,
                assistant_message_id=assistant_message_id,
            )
            if message is None or message.role != "assistant":
                raise SuggestedQuestionTargetNotFoundError("找不到目标助手消息")
            if not self._formal_text(message.content):
                raise SuggestedQuestionTargetInvalidError("目标助手消息没有正式正文")

            existing_questions = list(message.suggested_questions or [])
            revision = int(message.suggested_questions_revision or 0)
            status = message.suggested_questions_status or SUGGESTION_STATUS_IDLE
            if existing_questions and not force_refresh:
                self.db.rollback()
                return SuggestedQuestionRequestClaim(
                    claim=None,
                    questions=existing_questions,
                    message_id=message.id,
                    revision=revision,
                    status=status,
                )

            # 兼容旧前端：尚无结果时，即使后端自动任务 pending，也由这次请求领取
            # 更高 revision 并同步返回结果；迟到的自动任务会在 CAS 时被丢弃。
            claim = self._advance_revision(message)
            self.db.commit()
            return SuggestedQuestionRequestClaim(
                claim=claim,
                questions=existing_questions,
                message_id=message.id,
                revision=claim.revision,
                status=SUGGESTION_STATUS_PENDING,
            )
        except Exception:
            self.db.rollback()
            raise

    async def generate_claimed_questions(
        self,
        claim: SuggestedQuestionClaim,
        *,
        options: dict[str, Any] | None = None,
    ) -> SuggestedQuestionGenerationResult:
        """生成并按 revision CAS 写回；旧任务只能返回 applied=False。"""

        del options  # 预留兼容字段；辅助模型策略目前由服务端统一控制。
        try:
            dialog_content = self.build_dialog_content(claim.message_id)
            questions = await self._generate(dialog_content, claim.model_id)
            applied = self.store_generated_questions(claim=claim, questions=questions)
        except Exception:
            # 非 LLM 异常也必须释放 pending；CAS 保证已被更高 revision 抢占时不会误标失败。
            try:
                self.mark_generation_failed(claim)
            except Exception as mark_error:  # noqa: BLE001 — 保留原异常，日志只记类型
                logger.warning("标记推荐问题生成失败异常: error_type=%s", type(mark_error).__name__)
            raise
        if applied:
            return SuggestedQuestionGenerationResult(
                questions=questions,
                message_id=claim.message_id,
                revision=claim.revision,
                status=SUGGESTION_STATUS_READY,
                applied=True,
            )

        current = self._message_state(claim.message_id)
        return SuggestedQuestionGenerationResult(
            questions=current[0],
            message_id=claim.message_id,
            revision=current[1],
            status=current[2],
            applied=False,
        )

    def store_generated_questions(
        self,
        *,
        claim: SuggestedQuestionClaim,
        questions: list[str],
    ) -> bool:
        """仅当前 pending revision 可写入，避免迟到结果覆盖。"""

        try:
            updated = (
                self.db.query(MessageModel)
                .filter(
                    MessageModel.id == claim.message_id,
                    MessageModel.suggested_questions_revision == claim.revision,
                    MessageModel.suggested_questions_status == SUGGESTION_STATUS_PENDING,
                )
                .update(
                    {
                        "suggested_questions": list(questions),
                        "suggested_questions_status": SUGGESTION_STATUS_READY,
                    },
                    synchronize_session=False,
                )
            )
            self.db.commit()
            self.db.expire_all()
            return updated == 1
        except Exception:
            self.db.rollback()
            raise

    def mark_generation_failed(self, claim: SuggestedQuestionClaim) -> bool:
        """仅失败任务仍持有当前 revision 时更新状态，保留上一批问题。"""

        try:
            updated = (
                self.db.query(MessageModel)
                .filter(
                    MessageModel.id == claim.message_id,
                    MessageModel.suggested_questions_revision == claim.revision,
                    MessageModel.suggested_questions_status == SUGGESTION_STATUS_PENDING,
                )
                .update(
                    {"suggested_questions_status": SUGGESTION_STATUS_FAILED},
                    synchronize_session=False,
                )
            )
            self.db.commit()
            return updated == 1
        except Exception:
            self.db.rollback()
            raise

    def build_dialog_content(self, assistant_message_id: str) -> str:
        """只读取目标 assistant 及其之前最近一条 user，禁止漂移到后续轮次。"""

        target = self.db.query(MessageModel).filter(MessageModel.id == assistant_message_id).first()
        if target is None:
            return ""
        messages = (
            self.db.query(MessageModel)
            .filter(MessageModel.conversation_id == target.conversation_id)
            .order_by(
                MessageModel.sequence.asc().nullsfirst(),
                MessageModel.created_at.asc(),
                MessageModel.id.asc(),
            )
            .all()
        )
        target_index = next((index for index, item in enumerate(messages) if item.id == target.id), -1)
        if target_index < 0:
            return ""

        latest_ai = self._formal_text(messages[target_index].content)
        latest_user = ""
        for message in reversed(messages[:target_index]):
            if message.role != "user":
                continue
            latest_user = self._formal_text(message.content)
            if latest_user:
                break

        lines = []
        if latest_user:
            lines.append(f"用户: {latest_user}")
        if latest_ai:
            lines.append(f"助手: {latest_ai}")
        return "\n".join(lines)

    async def _generate(self, dialog_content: str, conversation_model_id: str) -> list[str]:
        if not dialog_content:
            return list(self.FALLBACK_QUESTIONS)
        try:
            prompt, prompt_metadata = prompt_manager.format_prompt_with_metadata(
                "generate_suggested_questions",
                content=dialog_content,
            )
            litellm_model, _, litellm_kwargs = self._resolve_utility_model(conversation_model_id)
            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=self.MAX_TOKENS,
                timeout=self.UTILITY_LLM_TIMEOUT,
                **merge_litellm_kwargs(
                    "suggest_questions",
                    litellm_kwargs,
                    prompt_metadata=prompt_metadata,
                ),
            )
            raw = response.choices[0].message.content or ""
            return ChatUtils.parse_questions(raw)[:3] or list(self.FALLBACK_QUESTIONS)
        except Exception as error:  # noqa: BLE001 — 推荐问题失败不能影响正文终态
            logger.warning("生成推荐问题失败，使用回退问题: error_type=%s", type(error).__name__)
            return list(self.FALLBACK_QUESTIONS)

    def _resolve_utility_model(self, conversation_model_id: str) -> tuple:
        try:
            return llm_manager.resolve_model(self.UTILITY_MODEL_ID)
        except ValueError:
            return llm_manager.resolve_model(conversation_model_id)

    def _advance_revision(self, message: MessageModel) -> SuggestedQuestionClaim:
        revision = int(message.suggested_questions_revision or 0) + 1
        message.suggested_questions_revision = revision
        message.suggested_questions_status = SUGGESTION_STATUS_PENDING
        return SuggestedQuestionClaim(
            message_id=message.id,
            conversation_id=message.conversation_id,
            revision=revision,
            model_id=message.model_id or message.conversation.model_id,
        )

    def _locked_message(
        self,
        *,
        assistant_message_id: str | None,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> MessageModel | None:
        query = self.db.query(MessageModel)
        if conversation_id is not None or user_id is not None:
            query = query.join(ConversationModel, ConversationModel.id == MessageModel.conversation_id)
        if conversation_id is not None:
            query = query.filter(MessageModel.conversation_id == conversation_id)
        if user_id is not None:
            query = query.filter(ConversationModel.user_id == str(user_id))
        query = query.filter(MessageModel.role == "assistant")
        if assistant_message_id is not None:
            query = query.filter(MessageModel.id == assistant_message_id)
        else:
            query = query.order_by(
                MessageModel.sequence.desc().nullslast(),
                MessageModel.created_at.desc(),
                MessageModel.id.desc(),
            )
        # 同一 Session 可能已经缓存过旧 revision；populate_existing 强制用加锁查询结果
        # 覆盖 identity map，PostgreSQL 的行锁才能真正串行化跨 worker 领取。
        return query.populate_existing().with_for_update().first()

    def _message_state(self, message_id: str) -> tuple[list[str], int, str]:
        self.db.expire_all()
        message = self.db.query(MessageModel).filter(MessageModel.id == message_id).first()
        if message is None:
            return [], 0, SUGGESTION_STATUS_FAILED
        return (
            list(message.suggested_questions or []),
            int(message.suggested_questions_revision or 0),
            message.suggested_questions_status or SUGGESTION_STATUS_IDLE,
        )

    @staticmethod
    def _formal_text(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                text = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                text = getattr(block, "text", None)
            if block_type == "text" and isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)


def claim_auto_suggested_questions(
    *,
    db: Session,
    assistant_message_id: str,
    run_id: str,
) -> SuggestedQuestionClaim | None:
    return SuggestedQuestionService(db).claim_auto_generation(
        assistant_message_id=assistant_message_id,
        run_id=run_id,
    )
