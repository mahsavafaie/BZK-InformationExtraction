import re
import logging

logger = logging.getLogger(__name__)

def extract_xml_info(sequence, tag_name, smallest_distance: bool = True, remove_tags_inside: bool = True, allow_partial_match: bool = False) -> str:
    """
    Extracts the information between the tags <s_tag_name> and </s_tag_name> from the sequence.
    If smallest_distance is True, the function will return the information between the closest tags.
    If tags_inside is 'remove', the tags will be removed from the extracted information.
    """
    start_tag = f"<s_{tag_name}>"
    end_tag = f"<\/s_{tag_name}>"

    possible_texts = []
    for start_tag_match in re.finditer(start_tag, sequence):
        for end_tag_match in re.finditer(end_tag, sequence):
            if start_tag_match.end() <= end_tag_match.start():
                possible_texts.append(sequence[start_tag_match.end():end_tag_match.start()])
    if len(possible_texts) > 0:
        selected_text = min(possible_texts, key=len) if smallest_distance else max(possible_texts, key=len)
        if remove_tags_inside:
            # Remove everything between < and >
            selected_text = re.sub(r'<[^>]*>', '', selected_text)
        return selected_text.strip()
    else:
        if not allow_partial_match:
            return ""
        # an open and close tag together in the right order does not exist
        # -> search for text left to a closing tag e.g. foo</s_tag_name> or right to an opening tag e.g. <s_tag_name>foo
        #    until the next tag is found
        possible_texts = []
        for start_tag_match in re.finditer(start_tag, sequence):
            end_pos = min(sequence.find('<', start_tag_match.end()), sequence.find('>', start_tag_match.end()))
            if end_pos == -1:
                end_pos = len(sequence)
            possible_texts.append(sequence[start_tag_match.end():end_pos])
        for end_tag_match in re.finditer(end_tag, sequence):
            start_pos = max(sequence.rfind('<', 0, end_tag_match.start()), sequence.rfind('>', 0, end_tag_match.start()))
            if start_pos == -1:
                start_pos = 0
            else:
                start_pos += 1
            possible_texts.append(sequence[start_pos:end_tag_match.start()])
        if len(possible_texts) == 0:
            return ""
        selected_text = min(possible_texts, key=len) if smallest_distance else max(possible_texts, key=len)
        return selected_text.strip()


def _split_only_if_not_inside_quotes(sequence: str, split_char: str) -> list:
    """
    Splits the sequence by the split_char if the split_char is not inside a quote.
    """
    splits = []
    in_quote = False
    buffer = ""
    for char in sequence:
        if char == '"':
            in_quote = not in_quote
        if char == split_char and not in_quote:
            buffer = buffer.strip()
            if buffer:
                splits.append(buffer)
            buffer = ""
        else:
            buffer += char
    buffer = buffer.strip()
    if buffer:
        splits.append(buffer)
    return splits

def extract_json_info(sequence) -> str:
    """
    Extracts the information from a json like structure.
    """
    # find begin and end of json
    start_pos = sequence.find("{")
    end_pos = sequence.rfind("}")
    if start_pos == -1:
        start_pos = 0
    if end_pos == -1:
        end_pos = len(sequence)
    
    # split each key value pair by newline or comma
    semi_json_sequence = sequence[start_pos:end_pos]
    # split by newline first
    json_info = semi_json_sequence.split("\n")
    # split by comma if newline does not work
    attribute_value_pairs = []
    for info in json_info:
        attribute_value_pairs.extend(_split_only_if_not_inside_quotes(info, ","))

    print(attribute_value_pairs)
    return_dict = {}
    for pair in attribute_value_pairs:
        one_attribute_value_pair = _split_only_if_not_inside_quotes(pair, ":")
        if len(one_attribute_value_pair) < 2:
            logger.warning(f"Could not split {pair} into key value pair")
            continue
        key = one_attribute_value_pair[0].strip()
        value = one_attribute_value_pair[1].strip()

        key = key.strip("\"' ")
        value = value.strip("\"' ")

        print(f"Key: {key}, Value: {value}")

        return_dict[key] = value
    return return_dict


        

    



