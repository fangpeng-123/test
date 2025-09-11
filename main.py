from langchain_ollama import OllamaLLM
ollm = OllamaLLM(model="qwen3:8b")
print(ollm.invoke("你好"))