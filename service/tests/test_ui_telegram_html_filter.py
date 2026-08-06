"""Tests for the telegram_html template filter — renders the small Telegram HTML
parse-mode subset used in `human` step prompt_templates as real HTML on the
/ui/approvals pages, without trusting arbitrary embedded content (e.g. {{summary}}
or LLM/agent output) enough to mark the whole message `| safe`."""
from src.ui.helpers import _telegram_html_to_safe_html


def _render(text: str) -> str:
    return str(_telegram_html_to_safe_html(text))


def test_bold_tag_renders_as_real_html():
    assert _render("<b>Approve?</b>") == "<b>Approve?</b>"


def test_code_and_italic_tags_render_as_real_html():
    assert _render("<code>run-123</code>") == "<code>run-123</code>"
    assert _render("<i>note</i>") == "<i>note</i>"
    assert _render("<em>note</em>") == "<em>note</em>"
    assert _render("<strong>note</strong>") == "<strong>note</strong>"


def test_https_link_renders_with_safe_target_attrs():
    result = _render('<a href="https://example.com/run/1">the run</a>')
    assert result == '<a href="https://example.com/run/1" target="_blank" rel="noopener noreferrer">the run</a>'


def test_http_link_also_allowed():
    result = _render('<a href="http://example.com">link</a>')
    assert 'href="http://example.com"' in result


def test_javascript_scheme_link_not_restored():
    # href doesn't match the https?:// requirement, so the <a> tag stays escaped text —
    # it must never become a live anchor with a javascript: URL.
    result = _render('<a href="javascript:alert(1)">click</a>')
    assert "<a href=" not in result
    assert "javascript:alert(1)" in result  # visible as inert text, not executed


def test_script_tag_stays_escaped_not_executed():
    result = _render("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_unrecognized_tag_stays_escaped():
    result = _render("<span onclick=\"evil()\">hi</span>")
    assert "<span" not in result
    assert "&lt;span" in result


def test_plain_text_passes_through_unescaped_for_safe_chars():
    assert _render("Approve remediation for api-gateway?") == "Approve remediation for api-gateway?"


def test_ampersand_and_quotes_in_plain_text_are_escaped():
    result = _render('Deploy "v2" & continue?')
    assert "<" not in result.replace("&lt;", "")  # no stray literal '<'
    assert "&#34;v2&#34;" in result or "&quot;v2&quot;" in result
    assert "&amp;" in result


def test_realistic_approval_test_message_renders_correctly():
    text = (
        "<b>VectorStep approval request</b>\n\n"
        "Pipeline: <code>approval-test</code>\n"
        "Run: <code>6d55bf1f-c231-42b1-9afd-4bb3271452d8</code>\n\n"
        "Pre-check: Pre-check complete — ready for human approval\n\n"
        "Approve to continue to the post-approval step, or reject to abort."
    )
    result = _render(text)
    assert "<b>VectorStep approval request</b>" in result
    assert "<code>approval-test</code>" in result
    assert "<code>6d55bf1f-c231-42b1-9afd-4bb3271452d8</code>" in result
    assert "&lt;" not in result  # every tag in this message was recognized and restored
