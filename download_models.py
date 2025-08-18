import nltk
from sentence_transformers import SentenceTransformer

def load_local_model(model_path):
    try:
        # 从本地路径加载模型
        model = SentenceTransformer(model_path)
        print(f"成功加载本地模型: {model_path}")
        return True
    except Exception as e:
        print(f"加载本地模型 {model_path} 失败: {str(e)}")
        return False

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
        print(f"资源 {resource_name} 已存在，无需下载")
    except LookupError:
        nltk.download(resource_name, quiet=True)
        print(f"成功下载nltk资源: {resource_name}")

if __name__ == "__main__":
    # 本地模型路径（确保与你实际下载的路径一致）
    local_models = [
        "./models/bge-small-zh",  # 对应BAAI/bge-small-zh
        "./models/all-MiniLM-L6-v2"  # 对应all-MiniLM-L6-v2
    ]
    
    for model_path in local_models:
        load_local_model(model_path)
    
    download_nltk_resource('punkt')
    
    print("所有资源加载完成")
    