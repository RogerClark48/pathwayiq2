import chromadb
import ollama
import anthropic

# Test Chroma
print("Testing Chroma...")
client = chromadb.Client()
collection = client.create_collection("test")
print("  Chroma OK")

# Test Ollama embedding
print("Testing Ollama embedding...")
response = ollama.embeddings(model="nomic-embed-text", prompt="test sentence")
dims = len(response["embedding"])
print(f"  Ollama OK — embedding dimensions: {dims}")

# Test Anthropic (just checks the client loads, not an API call)
print("Testing Anthropic client...")
client = anthropic.Anthropic()
print("  Anthropic client loaded OK")

print("\nAll systems go.")