def print_json_to_depth(data, depth=1, indent=0):
    """Recursively print a JSON-like dictionary to a specified depth.

    Args:
        data: The data structure to print.
        depth: Maximum depth to print (default 1).
        indent: Current indentation level (for recursive calls).
    """
    indent_str = '  ' * indent

    if depth <= 0:
        print(f"{indent_str}...")
        return

    if isinstance(data, dict):
        print(f"{indent_str}{{")
        for key, value in data.items():
            print(f"{indent_str}  {key}: ", end='')
            if isinstance(value, (dict, list)):
                print()
                print_json_to_depth(value, depth - 1, indent + 1)
            else:
                print(value)
        print(f"{indent_str}}}")

    elif isinstance(data, list):
        print(f"{indent_str}[")
        for item in data:
            if isinstance(item, (dict, list)):
                print_json_to_depth(item, depth - 1 if depth > 1 else depth, indent + 1)
            else:
                print(f"{indent_str}  {item}")
        print(f"{indent_str}]")

    else:
        print(f"{indent_str}{data}")
