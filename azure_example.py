import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

os.environ["AZURE_INFERENCE_CREDENTIAL"] = ""
api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", '')
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# Create the client (make sure the endpoint is correct for your subscription)
client = ChatCompletionsClient(
    endpoint='https://Phi-3-5-MoE-instruct-gaqxj.eastus2.models.ai.azure.com',
    credential=AzureKeyCredential(api_key)
)

# Get model info (for debugging purposes)
model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)


def organize_text(client, input_text):
    """
    Given a text block, use the language model to "organize" it.
    The prompt instructs the LM to remove unnecessary characters and line breaks,
    rewriting the text to be more readable without altering words.
    """
    messages = [
        {
            "role": "system",
            "content": "You need to only organize the information here since there are unnecessary characters and break lines. Rewrite the text without modifying any words, just making it readable and organized."
        },
        {
            "role": "user",
            "content": f"This is the text you need to rewrite: {input_text}"
        }
    ]
    payload = {
        "messages": messages,
        "max_tokens": 3000,
        "temperature": 0.2,
        "top_p": 0.1,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
    response = client.complete(payload)
    answer = response.choices[0].message.content
    return answer


def organize_document(client, parsed_text):
    """
    Process a parsed document (a dict with keys "case_info", "summary", and "sections")
    and organize each text block using the language model.
    Returns a new dictionary that is ready to be stored as JSON.
    """
    organized_doc = {}
    # Process top-level case info and summary.
    organized_doc["case_info"] = organize_text(client, parsed_text.get("case_info", ""))
    print(organized_doc["case_info"])
    organized_doc["summary"] = organize_text(client, parsed_text.get("summary", ""))
    print(organized_doc["summary"])
    # Process each section. For each key in the "sections" dictionary, run the organizing function.
    organized_doc["sections"] = {}
    sections = parsed_text.get("sections", {})
    for section_heading, content in sections.items():
        # Only process non-empty content.
        if content.strip() :
            organized_doc["sections"][section_heading] = organize_text(client, content)
        else:
            organized_doc["sections"][section_heading] = ""
    return organized_doc

    
# Organize the parsed document using the language model.
organized = organize_document(client, parsed_text)
    
# Print the resulting JSON in pretty format.
print("Organized Document:")
print(json.dumps(organized, indent=4))


os.environ["AZURE_INFERENCE_CREDENTIAL"] = ""
api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", '')
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")
    
client = ChatCompletionsClient(
    endpoint='https://Phi-3-5-MoE-instruct-gaqxj.eastus2.models.ai.azure.com',
    credential=AzureKeyCredential(api_key)
)


model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)

payload = {
  "messages": [
    {
      "role": "user",
      "content": "I am going to Paris, what should I see?"
    },
    {
      "role": "assistant",
      "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."
    },
    {
      "role": "user",
      "content": "What is so great about #1?"
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.8,
  "top_p": 0.1,
  "presence_penalty": 0,
  "frequency_penalty": 0
}
response = client.complete(payload)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Usage:")
print("	Prompt tokens:", response.usage.prompt_tokens)
print("	Total tokens:", response.usage.total_tokens)
print("	Completion tokens:", response.usage.completion_tokens)