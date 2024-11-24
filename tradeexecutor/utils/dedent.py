"""Helpers to deal with formatting text messages in decide_trades() diagnostics output"""

import re


def strip_except_newlines(text):
    # Split by newlines, strip each line, then rejoin
    return '\n'.join(line.strip() for line in text.splitlines())

def dedent_any(text: str) -> str:
    """Dedent variable indents of the text"""
    return re.sub(r'^\s+', '', strip_except_newlines(text), flags=re.MULTILINE)