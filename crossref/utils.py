def print_json_to_depth(data, depth=1, indent=0):
    """
    Recursively print a JSON-like dictionary to a specified depth.
    
    Args:
        data (dict/list/any): The data structure to print
        depth (int): Maximum depth to print (default is 1)
        indent (int): Current indentation level (for recursive calls)
    """
    # Determine the indentation string
    indent_str = '  ' * indent
    
    # Handle different types of data
    if depth <= 0:
        print(f"{indent_str}...")
        return
    
    if isinstance(data, dict):
        print(f"{indent_str}{{")
        for key, value in data.items():
            print(f"{indent_str}  {key}: ", end='')
            
            # Recursively print nested structures
            if isinstance(value, (dict, list)):
                print()
                print_json_to_depth(value, depth - 1, indent + 1)
            else:
                print(value)
        print(f"{indent_str}}}")
    
    elif isinstance(data, list):
        print(f"{indent_str}[")
        for item in data:
            # Recursively print nested structures
            if isinstance(item, (dict, list)):
                print_json_to_depth(item, depth - 1 if depth > 1 else depth, indent + 1)
            else:
                print(f"{indent_str}  {item}")
        print(f"{indent_str}]")
    
    else:
        # For simple types, just print directly
        print(f"{indent_str}{data}")