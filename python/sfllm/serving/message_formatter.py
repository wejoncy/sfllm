from typing import List, Dict, Any

def format_chat_messages(messages: List) -> str:
    """
    Format messages into a prompt with processed images.
    
    Args:
        messages: List of message objects with role and content
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                prompt += f"<start_of_turn>system\n{msg.content}<end_of_turn>\n"
            else:
                # Handle content list for system message
                system_content = ""
                for item in msg.content:
                    if item.type == "text":
                        system_content += f"{item.text} "
                prompt += f"<start_of_turn>system\n{system_content.strip()}<end_of_turn>\n"
        
        elif msg.role == "user":
            prompt += "<start_of_turn>user\n"
            if isinstance(msg.content, str):
                prompt += f"{msg.content}"
            else:
                # Handle content list for user message
                for item in msg.content:
                    if item.type == "text":
                        prompt += f"{item.text} "
                    else:
                        assert False, "Image URLs are not supported in this version"
            prompt += "<end_of_turn>\n"
        
        elif msg.role == "assistant":
            if isinstance(msg.content, str):
                prompt += f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"
            else:
                # Handle content list for assistant message
                asst_content = ""
                for item in msg.content:
                    if item.type == "text":
                        asst_content += f"{item.text} "
                prompt += f"<start_of_turn>model\n{asst_content.strip()}<end_of_turn>\n"
    
    prompt += "<start_of_turn>model\n"
    return prompt
