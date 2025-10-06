import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# 🔑 Set API key
os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


search_tool = DuckDuckGoSearchRun()

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 4: Ask research question
topic = "CBIT college in gandipet"
response = agent.run(f"Do a deep research on {topic} and report it.")
print("\n📌 Research Summary:\n")
print(response)
