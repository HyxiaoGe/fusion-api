import unittest

from app.services.mcp.provider_profiles import (
    CONTEXT7_READ_ONLY_TOOL_ALLOWLIST,
    endpoint_auth_binding_is_allowed,
    endpoint_tool_allowlist,
    endpoint_tool_guidance,
    endpoint_tool_schema_override,
    tool_is_allowed_for_endpoint,
)


class McpProviderProfileTests(unittest.TestCase):
    def test_context7_official_endpoint_has_fixed_tool_allowlist(self):
        endpoint = "https://MCP.CONTEXT7.COM./mcp"

        self.assertEqual(
            endpoint_tool_allowlist(endpoint),
            CONTEXT7_READ_ONLY_TOOL_ALLOWLIST,
        )
        self.assertTrue(tool_is_allowed_for_endpoint(endpoint, "resolve-library-id"))
        self.assertTrue(tool_is_allowed_for_endpoint(endpoint, "query-docs"))
        self.assertFalse(tool_is_allowed_for_endpoint(endpoint, "future-write-tool"))

    def test_context7_tools_have_trusted_guidance_and_fixed_schemas(self):
        endpoint = "https://mcp.context7.com/mcp"

        resolve_guidance = endpoint_tool_guidance(endpoint, "resolve-library-id")
        query_guidance = endpoint_tool_guidance(endpoint, "query-docs")
        self.assertIn("公开", resolve_guidance)
        self.assertIn("不得发送", resolve_guidance)
        self.assertIn("resolve-library-id", query_guidance)
        self.assertIn("必须先调用", query_guidance)
        self.assertIn("不得发送", query_guidance)

        self.assertEqual(
            endpoint_tool_schema_override(endpoint, "resolve-library-id"),
            {
                "type": "object",
                "properties": {
                    "libraryName": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                    },
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                "required": ["libraryName", "query"],
                "additionalProperties": False,
            },
        )
        self.assertEqual(
            endpoint_tool_schema_override(endpoint, "query-docs"),
            {
                "type": "object",
                "properties": {
                    "libraryId": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 200,
                    },
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                "required": ["libraryId", "query"],
                "additionalProperties": False,
            },
        )

    def test_non_profiled_endpoint_has_no_trusted_guidance_or_schema_override(self):
        endpoint = "https://example.com/mcp"

        self.assertIsNone(endpoint_tool_allowlist(endpoint))
        self.assertEqual(endpoint_tool_guidance(endpoint, "query-docs"), "")
        self.assertIsNone(endpoint_tool_schema_override(endpoint, "query-docs"))

    def test_official_credentials_are_bound_to_their_expected_endpoint_and_auth_shape(self):
        self.assertTrue(
            endpoint_auth_binding_is_allowed(
                "https://mcp.context7.com/mcp",
                auth_type="header",
                auth_name="CONTEXT7_API_KEY",
                credential_ref="CONTEXT7_API_KEY",
            )
        )
        self.assertTrue(
            endpoint_auth_binding_is_allowed(
                "https://mcp.context7.com/mcp",
                auth_type="none",
                auth_name=None,
                credential_ref=None,
            )
        )
        self.assertFalse(
            endpoint_auth_binding_is_allowed(
                "https://mcp.amap.com/mcp",
                auth_type="header",
                auth_name="CONTEXT7_API_KEY",
                credential_ref="CONTEXT7_API_KEY",
            )
        )
        self.assertFalse(
            endpoint_auth_binding_is_allowed(
                "https://mcp.context7.com/mcp",
                auth_type="header",
                auth_name="X-API-Key",
                credential_ref="CONTEXT7_API_KEY",
            )
        )
        self.assertTrue(
            endpoint_auth_binding_is_allowed(
                "https://mcp.amap.com/mcp",
                auth_type="query",
                auth_name="key",
                credential_ref="AMAP_MCP_API_KEY",
            )
        )
        self.assertFalse(
            endpoint_auth_binding_is_allowed(
                "https://mcp.context7.com/mcp",
                auth_type="query",
                auth_name="key",
                credential_ref="AMAP_MCP_API_KEY",
            )
        )


if __name__ == "__main__":
    unittest.main()
