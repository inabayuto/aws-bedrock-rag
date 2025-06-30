# 必要なPythonライブラリをインポート
import boto3
import json
import numpy as np
import faiss

# AWS SDK for Pythonで、Bedrock用のAPIクライアントを作成
client = boto3.client("bedrock-runtime", region_name="us-east-1")  # リージョンを明示

# 生成AIモデルを設定
llm = "amazon.titan-text-premier-v1:0"
embedding_model = "cohere.embed-multilingual-v3"

# 社内データを読み込む
docs = [
    {
        "id": 0,
        "title": "週次ミーティングの議事録",
        "content": "2025年6月28日に開催された週次ミーティングでは、今四半期のKPI進捗確認と、新規プロジェクト『HELLO-AI』の概要共有が行われました。"
    },
    {
        "id": 1,
        "title": "社内ハッカソン開催のお知らせ",
        "content": "HELLO社では7月15日に社内ハッカソンを開催します。参加希望者は7月5日までにGoogleフォームより申し込みをお願いします。"
    },
    {
        "id": 2,
        "title": "新入社員オンボーディング資料",
        "content": "この資料では、HELLO社の組織体制、業務フロー、利用ツール（Slack、Notion、Zoomなど）について説明しています。新入社員は必ず目を通してください。"
    }
]

# 社内文書をベクトル化
print("ベクトル化中...")

# ドキュメントをベクトル化
embeddings = []
for doc in docs:
    print(f"ドキュメント: {doc['title']}")

    try:
        response = client.invoke_model(
            modelId=embedding_model,
            body=json.dumps({
                "texts": [doc["content"]],
                "input_type": "search_document"
            }),
        )
        response_body = json.loads(response["body"].read())
        embedding = np.array(response_body["embeddings"][0])
        embeddings.append(embedding)
        # print(embeddings)
    except Exception as e:
        print(f"エラー発生: {e}")

# NumPy配列に変換
embeddings_array = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings_array)

print(embeddings_array.shape)

# FAISSインデックスを作成
dimension = embeddings_array.shape[1] # ベクトルの次元数（1024） 
index = faiss.IndexFlatIP(dimension) # 内積（コサイン類似度）でインデックス計算
index.add(embeddings_array) # ベクトル追加


# 検索実行
query = "新規プロジェクトの名前は？"

# クエリをベクトル化
response = client.invoke_model(
    modelId=embedding_model,
    body=json.dumps({
        "texts": [query],
        "input_type": "search_query"
    })
)
response_body = json.loads(response["body"].read())

# NumPy配列に変換し、ベクトルを正規化
query_embedding = np.array(response_body["embeddings"][0]).reshape(1, -1).astype('float32')
faiss.normalize_L2(query_embedding)

# コサイン類似度で検索（上位1件を取得）
similarities, indices = index.search(query_embedding, 1)

retrieved_docs = []
for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
    doc = docs[idx]
    retrieved_docs.append(doc)

context = retrieved_docs[0]['content']

# 推論実行
prompt = f"""
あなたはHELLO社のAIエージェントです。
以下のドキュメントを参考に、ユーザーの質問に回答してください。
ドキュメント: {context}
ユーザーの質問: {query}
"""
print(prompt)
print("推論実行中...")
try:
    # LLMに推論を実行
    response = client.invoke_model(
        modelId=llm,
        body=json.dumps({
            "inputText": prompt
        })
    )
    response_body = json.loads(response["body"].read())
    print("推論結果：", response_body["results"][0]["outputText"])
except Exception as e:
    print(f"エラー発生: {e}")

print("推論完了")



