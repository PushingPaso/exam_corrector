"""
Entry point for running the MCP server.
"""

import sys
import asyncio
from pathlib import Path
from exam.mcp import run_mcp_server


def main():
    """Main entry point."""
    # Get exam directory from command line or use None
    exam_dir = None
    if len(sys.argv) > 1:
        exam_dir = Path(sys.argv[1])
        if not exam_dir.exists():
            print(f"Error: Directory {exam_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"# MCP Server: Using exam directory: {exam_dir}", file=sys.stderr)
    else:
        print("# MCP Server: No exam directory specified (some tools will be limited)", file=sys.stderr)
    
    print("# MCP Server: Starting...", file=sys.stderr)
    print("# MCP Server: Available tools:", file=sys.stderr)
    print("#   - list_questions", file=sys.stderr)
    print("#   - get_question", file=sys.stderr)
    print("#   - list_students", file=sys.stderr)
    print("#   - read_student_answer", file=sys.stderr)
    print("#   - get_checklist", file=sys.stderr)
    print("#   - assess_feature", file=sys.stderr)
    print("#   - calculate_score", file=sys.stderr)
    print("#   - search_course_material", file=sys.stderr)
    print("#   - generate_feedback", file=sys.stderr)
    print("#   - save_assessment", file=sys.stderr)
    print("# MCP Server: Ready", file=sys.stderr)
    
    # Run the server
    try:
        asyncio.run(run_mcp_server(exam_dir))
    except KeyboardInterrupt:
        print("\n# MCP Server: Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"# MCP Server: Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()