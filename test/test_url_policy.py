"""URL Policy 单元测试。"""


def test_url_policy_rejects_private_hosts():
    from app.services.security.url_policy import evaluate_url_policy

    for url in [
        "http://localhost/admin",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.5/private",
        "http://example.local/page",
    ]:
        result = evaluate_url_policy(url)
        assert not result.allowed
        assert result.reason == "private_host"


def test_url_policy_rejects_credentials_and_sensitive_query():
    from app.services.security.url_policy import evaluate_url_policy

    credential_result = evaluate_url_policy("https://user:pass@example.com/page")
    assert not credential_result.allowed
    assert credential_result.reason == "credentials_in_url"

    token_result = evaluate_url_policy("https://example.com/page?token=abc&safe=1")
    assert not token_result.allowed
    assert token_result.reason == "sensitive_query"
    assert token_result.safe_log_url == "https://example.com/page"


def test_url_policy_allows_public_http_url_and_sanitizes_log_url():
    from app.services.security.url_policy import evaluate_url_policy

    result = evaluate_url_policy("https://Example.com:443/a/../b?q=ok#fragment")

    assert result.allowed
    assert result.normalized_url == "https://example.com/b?q=ok"
    assert result.safe_log_url == "https://example.com/b"


def test_url_policy_rejects_invalid_port_without_raising():
    from app.services.security.url_policy import evaluate_url_policy

    result = evaluate_url_policy("https://example.com:bad/page")

    assert not result.allowed
    assert result.reason == "invalid_host"
