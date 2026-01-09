import sys
import json
from pathlib import Path
from inspect_ai.log import read_eval_log


def escape_latex(text):
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '_': r'\_',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    return result


def format_content(content):
    if isinstance(content, str):
        return escape_latex(content)
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(escape_latex(text))
                elif item.get("type") == "reasoning":
                    reasoning = item.get("reasoning", "")
                    if reasoning:
                        parts.append(r"\textit{[Reasoning]} " + escape_latex(reasoning))
                elif item.get("type") == "tool_use":
                    parts.append(f"[Tool Use: {escape_latex(item.get('name', 'unknown'))}]")
                elif item.get("type") == "tool_result":
                    content_str = str(item.get('content', ''))
                    truncated = content_str[:500] + ("..." if len(content_str) > 500 else "")
                    parts.append(f"[Tool Result: {escape_latex(truncated)}]")
            elif hasattr(item, '__dict__'):
                item_type = getattr(item, 'type', None)
                if item_type == 'text':
                    text = getattr(item, 'text', '')
                    if text:
                        parts.append(escape_latex(text))
                elif item_type == 'reasoning':
                    reasoning = getattr(item, 'reasoning', '')
                    if reasoning:
                        parts.append(r"\textit{[Reasoning]} " + escape_latex(reasoning))
                else:
                    item_str = str(item)
                    if len(item_str) < 200:
                        parts.append(escape_latex(item_str))
            else:
                parts.append(escape_latex(str(item)))
        return "\n\n".join(parts)
    else:
        return escape_latex(str(content))


def format_json_for_latex(obj):
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except:
            pass

    json_str = json.dumps(obj, indent=2) if not isinstance(obj, str) else obj

    if len(json_str) > 500:
        json_str = json_str[:500] + "\n... (truncated)"

    return escape_latex(json_str)


def render_transcript_to_latex(eval_path, output_path, standalone=False):
    eval_path = Path(eval_path)
    output_path = Path(output_path)

    print(f"Loading eval: {eval_path}")
    log = read_eval_log(str(eval_path))

    if not log.samples:
        print("No samples found in eval")
        return

    sample = log.samples[0]
    print(f"Rendering sample {sample.id} with {len(sample.messages)} messages")

    latex_lines = []

    if standalone:
        latex_lines.append(r"\documentclass{article}")
        latex_lines.append(r"\usepackage{transcript}")
        latex_lines.append(r"\begin{document}")
        latex_lines.append(r"")

    model_name = getattr(log.eval, "model", "Unknown") if hasattr(log, "eval") else "Unknown"
    task_name = getattr(log.eval, "task", "Unknown") if hasattr(log, "eval") else "Unknown"

    latex_lines.append(r"\begin{evaltranscript}{" + escape_latex(model_name) + r"}{Sample " + str(sample.id) + r"}")
    latex_lines.append(r"")

    pending_tool_calls = {}

    for msg in sample.messages:
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")

        if role == "system":
            formatted_content = format_content(content)
            latex_lines.append(r"\systemmsg{")
            latex_lines.append(formatted_content)
            latex_lines.append(r"}")
            latex_lines.append(r"")

        elif role == "user":
            formatted_content = format_content(content)
            latex_lines.append(r"\usermsg{")
            latex_lines.append(formatted_content)
            latex_lines.append(r"}")
            latex_lines.append(r"")

        elif role == "assistant":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = getattr(tc, "id", None)
                    tc_function = tc.function if hasattr(tc, "function") else "unknown"
                    tc_args = tc.arguments if hasattr(tc, "arguments") else {}

                    pending_tool_calls[tc_id] = tc_function

                    formatted_args = format_json_for_latex(tc_args)
                    latex_lines.append(r"\toolcallmsg{" + escape_latex(tc_function) + r"}{")
                    latex_lines.append(r"\textbf{Arguments:}")
                    latex_lines.append(r"")
                    latex_lines.append(formatted_args)
                    latex_lines.append(r"}")
                    latex_lines.append(r"")

            if content:
                formatted_content = format_content(content)
                latex_lines.append(r"\assistantmsg{")
                latex_lines.append(formatted_content)
                latex_lines.append(r"}")
                latex_lines.append(r"")

        elif role == "tool":
            tc_id = getattr(msg, "tool_call_id", None)
            tool_function = getattr(msg, "function", None)

            if not tool_function and tc_id in pending_tool_calls:
                tool_function = pending_tool_calls[tc_id]

            if not tool_function:
                tool_function = "unknown"

            formatted_content = format_content(content)

            if len(formatted_content) > 2000:
                formatted_content = formatted_content[:2000] + "\n\n... (truncated for length)"

            latex_lines.append(r"\toolmsg{" + escape_latex(tool_function) + r"}{")
            latex_lines.append(formatted_content)
            latex_lines.append(r"}")
            latex_lines.append(r"")

            if tc_id in pending_tool_calls:
                del pending_tool_calls[tc_id]

    latex_lines.append(r"\end{evaltranscript}")

    if standalone:
        latex_lines.append(r"")
        latex_lines.append(r"\end{document}")

    output_content = "\n".join(latex_lines)

    print(f"Writing LaTeX to: {output_path}")
    output_path.write_text(output_content, encoding="utf-8")

    if standalone:
        print(f"Done! Compile with: pdflatex {output_path.name}")
    else:
        print(f"Done! Include in your paper with: \\input{{{output_path.name}}}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_latex_transcript.py <eval_file.eval> [output.tex] [--standalone]")
        print("")
        print("By default, generates a snippet for inclusion in an existing document.")
        print("Use --standalone to generate a complete compilable document.")
        sys.exit(1)

    eval_file = Path(sys.argv[1])

    standalone = "--standalone" in sys.argv
    args_without_flags = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if len(args_without_flags) > 1:
        output_file = Path(args_without_flags[1])
    else:
        output_file = Path("transcript.tex")

    render_transcript_to_latex(eval_file, output_file, standalone=standalone)
