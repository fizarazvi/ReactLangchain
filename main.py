from dotenv import load_dotenv
load_dotenv()

from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    return len(text)

# Create a tool object with the required metadata
text_length_tool = Tool(
    func=get_text_length,
    name="get_text_length",
    description="Returns the length of the text by characters"
)

if __name__ == "__main__":
    print("Hello reAct Langchain")
    tools = [text_length_tool]
    template = """
    Answer the following questions as best as you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input of the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=",".join([t.name for t in tools])
    )
    
    llm = ChatOpenAI(temperature=0, model_kwargs={"stop": ["\nObservation", "Observation"]})

    agent = {"input": lambda x: x["input"]} | prompt | llm
    res = agent.invoke({"input": "What is the text length of 'DOG' in characters?"})
    print(res)
