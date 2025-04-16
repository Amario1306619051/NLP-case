from openai import OpenAI
from dotenv import load_dotenv
import os

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= os.getenv('SECRET_KEY'),
)

def humanize(question, answer, distance):

    if min(distance) <= 450:
        completion = client.chat.completions.create(
            extra_headers={
                # Optional: Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model="google/gemini-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah asisten virtual perusahaan Nawatech yang menjawab dalam Bahasa Indonesia. "
                        "Anda akan menerima satu pertanyaan dari pengguna dan dua potongan jawaban dalam bentuk list. "
                        "Jawaban Anda **hanya boleh berdasarkan informasi dari dua jawaban tersebut**, tanpa menambahkan informasi lain dari luar atau mengarang bebas. "
                        "Jika pertanyaan pengguna relevan dengan salah satu atau kedua jawaban, susun respons dalam bentuk paragraf yang singkat, jelas, dan sopan. "
                        "Namun jika pertanyaannya tidak berkaitan dengan jawaban yang diberikan, katakan dengan jujur bahwa Anda tidak memiliki informasi yang cukup, "
                        "dan minta pengguna untuk mengajukan pertanyaan yang lebih spesifik. "
                        "Jawaban Anda tidak boleh terpotong, tidak boleh menambahkan opini pribadi, dan tidak boleh menyimpang dari data yang tersedia."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Diberikan pertanyaan: {question}.\n"
                        f"Dan berikut adalah dua jawaban dalam bentuk list:\n{answer}\n\n"
                        "Silakan berikan jawaban sesuai instruksi."
                    )
                }
            ],
            max_tokens=400,
            temperature=0.5  # lebih rendah untuk menghindari halusinasi
        )
    else:
        completion = client.chat.completions.create(
            extra_headers={
                # Optional: Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model="google/gemini-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah asisten virtual yang ramah dan komunikatif. "
                        "Tugas Anda adalah menjawab pertanyaan dalam Bahasa Indonesia, dengan gaya seperti teman bicara. "
                        "Meskipun sebagian besar pertanyaan mungkin tidak berkaitan langsung dengan Nawatech, "
                        "tetaplah menjawab dengan sopan dan informatif. "
                        "Di akhir setiap jawaban, ajak secara halus pengguna untuk bertanya seputar Nawatech, "
                        "tanpa terdengar memaksa atau terlalu formal. "
                        "Contoh ajakan: 'Ngomong-ngomong, penasaran nggak sih soal Nawatech? 😊'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Diberikan pertanyaan: {question}. "
                            "Tolong jawab dengan gaya santai dan ramah. "
                            "Akhiri dengan mengajak pengguna untuk bertanya tentang Nawatech."
                }
            ],
            max_tokens=200,
            temperature=0.7
        )

    return completion.choices[0].message.content

def define_paragraph(question, answer):
    completion = client.chat.completions.create(
        model="google/gemini-pro",
        messages=[
            {
                "role": "system",
                "content": f"Anda adalah asisten yang menjawab dalam Bahasa Indonesia. Diberikan pertanyaan : {question} dan jawaban : {answer} tentang perusahaan Nawatech"
            },
            {
                "role": "user",
                "content": (
                    "Kombinasikan pertanyaan dan jawaban tersebut menjadi satu kalimat yang jelas."
                    "Gunakan bahasa formal dan eksplisit"
                    "hanya boleh menggunakan informasi (fakta) yang diberikan pada jawaban dan pertanyaan"
                    "Gunakan maksimal 20 kata"
                    "hindari karakter seperti /n dan lain lain"
                )
            }
        ],
        max_tokens=200,
        temperature=0.7
    )
    return completion.choices[0].message.content

def embbed_online(text):
    completion = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return completion.data[0].embedding