import xml.etree.ElementTree as ET

# Function to recursively generate the XPath for a given element
def get_xpath(element, root):
    path_parts = []
    while element != root:
        # Get the element's tag, which may include namespaces
        tag = element.tag
        parent = find_parent(root, element)
        
        # Check if there are siblings with the same tag
        if parent is not None:
            same_tag_siblings = [e for e in parent if e.tag == element.tag]
            if len(same_tag_siblings) > 1:
                # If there are multiple siblings with the same tag, add the index (1-based) in the XPath
                index = same_tag_siblings.index(element) + 1
                path_parts.append(f"{tag}[{index}]")
            else:
                path_parts.append(tag)
        else:
            path_parts.append(tag)
        element = parent
    return "/".join(reversed(path_parts))

# Function to find the parent of an element
def find_parent(root, child):
    for parent in root.iter():
        if child in parent:
            return parent
    return None

# Function to find the element and return its XPath
def find_element_and_get_xpath(file_path, search_text, namespaces):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Iterate through all elements in the tree and search for the desired entry
    for elem in root.iter():
        if elem.text and search_text in elem.text:
            # Return the XPath for the element
            return get_xpath(elem, root)
    return None

# Define the path to the XML file
file_path = './dcrm-003(1).xml'
# Define namespaces used in the XML (if applicable)
namespaces = {
    'drmd': 'https://example.org/drmd',
    'dcc': 'https://ptb.de/dcc',
    'si': 'https://ptb.de/si'
}

# Search for the element with the specific text and return its XPath
search_text = "manufacturer"
xpath_result = find_element_and_get_xpath(file_path, search_text, namespaces)

if xpath_result:
    print(f"XPath for '{search_text}': {xpath_result}")
else:
    print(f"Element with text '{search_text}' not found.")

