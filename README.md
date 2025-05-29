# AG-Pipeline-para-IA-Generativa-no-Lakehouse-Databricks

Vamos trabalhar  com um  dos  temas  mais  quentes  do  momento  no universo da Inteligência Artificial e implementar um RAG Pipeline com LLM do Databricks.

A Geração Aumentada por Recuperação (RAG) é uma técnica que aprimora modelos de linguagem de grande escala (LLMs) ao integrá-los com fontes externas de conhecimento. Ao  receber  uma  consulta,  o  sistema  busca  informações  relevantes  em  bases  de  dados atualizadas. Essas informações são então combinadas com a entrada original e processadas pelo modelo de linguagem para gerar uma resposta mais precisa e contextualizada. Essa  abordagem  oferece  diversas vantagens,  como  a  capacidade  de  fornecer  respostas atualizadas  sem  a  necessidade  de  retreinar  o  modelo,  redução  de  "alucinações"  (respostas incorretas geradas pelo modelo) e maior controle sobre as fontes de informação utilizadas. O  RAG  é  especialmente  útil  em  aplicações  que  exigem  informações  atualizadas  ou específicas  de  um  domínio,  como  chatbots  de  suporte  ao  cliente  ou  assistentes  virtuais  em setores especializados.

# Relação entre RAG e IA Generativa

A   relação   entre   RAG   (Retrieval   Augmented   Generation)   e   IA   Generativa   é   de complementaridade. Ambos são componentes que, juntos, formam uma abordagem poderosa para  gerar  respostas  mais  informadas,  contextuais  e  confiáveis  em  sistemas  de  Inteligência Artificial.IA  Generativa:  Modelos  de  IA  Generativa,  como  os  LLMs  (Large  Language  Models), são treinados  para  gerar  texto  com  base  em  padrões  aprendidos  a  partir  de  grandes  volumes  de dados. Embora sejam muito eficazes em tarefas de linguagem natural, eles têm limitações, como a possibilidade de "alucinações" (gerar informações erradas ou fictícias) e dependência de dados de treinamento que podem estar desatualizados.RAG  como  Solução  para  Limitações:  A  técnica  de  RAG  integra  a  IA  Generativa  com mecanismos  de  recuperação  de  informações.  Quando  uma  consulta  é  feita,  o  sistema  busca informações relevantes em bases de dados externas ou específicas (como documentos, bancos de  conhecimento  ou  até  mesmo  a  web).  Essas  informações  são  então  usadas  como  contexto adicional para a IA Generativa, melhorando a precisão e relevância das respostas.

🛠 Funcionamento Combinado:

* O  mecanismo  de  recuperação  (retrieval)  identifica  dados  relevantes  em  tempo real a partir de um banco de dados vetorial.

* A  IA  Generativa  processaesses  dados  junto  com  a  entrada  original  do  usuário, gerando uma resposta que combina o conhecimento do modelo com informações recuperadas.

🛠 Benefícios da Integração:

* Atualização  Contínua:  O  RAG  permite  que  os  modelos  generativos  acessem informações atualizadas sem necessidade de serem retreinados.

* Maior  Confiabilidade:  As  respostas  são  mais  embasadas  em  fontes  externas verificáveis, reduzindo erros.

* Adaptabilidade:  É  possível  personalizar  o  sistema  para  diferentes  domínios  ou necessidades específicas.

Exemplo  Prático:  Imagine  um  chatbot  que  responde  perguntas  sobre  legislação.  Sem  o RAG,  ele  pode  fornecer  respostas  desatualizadas  baseadas  apenas  nos  dados  de  treinamento. Com  o  RAG,  o  sistema  pode  consultar  fontes  legais  atualizadas  antes  de  gerar  uma  resposta, garantindo maior precisão.

Neste projeto vamos construir um RAG Pipeline para uma aplicação de IA Generativa. Usamos um LLM disponibilizado na plataforma Databricks e executaremos o Lab no Google Colab. Todos os arquivos estão disponível.

# Por que usar RAG?

Usar RAG (Retrieval-Augmented Generation) em uma aplicação de IA Generativa tem como principal objetivo aumentar a precisão, a confiabilidade e a atualidade das respostas geradas por modelos de linguagem (LLMs). Aqui estão os principais motivos pelos quais o uso do RAG é extremamente valioso:

🎯 1. Respostas mais precisas e baseadas em fatos
Problema: Modelos LLMs podem gerar alucinações, ou seja, inventar informações que parecem corretas, mas são falsas.

Solução com RAG: O sistema busca documentos reais e confiáveis (por exemplo, PDFs, bases jurídicas, artigos técnicos) e fornece essas informações como contexto adicional ao modelo. Isso aumenta a veracidade da resposta.

🔄 2. Informações sempre atualizadas
Problema: LLMs são treinados com dados de corte (ex: até 2023) e não têm acesso a novas informações.

Solução com RAG: O pipeline pode consultar fontes atualizadas (como bancos de dados vetoriais com documentos novos) em tempo real, sem a necessidade de retreinar o modelo.

🧠 3. Personalização por domínio
Problema: LLMs genéricos não têm conhecimento detalhado de domínios específicos (como direito tributário, medicina ou engenharia).

Solução com RAG: O sistema pode ser alimentado com dados de domínio específicos, como documentos internos de uma empresa ou artigos científicos, adaptando o comportamento do modelo a contextos muito específicos.

✅ 4. Controle das fontes de informação
Com RAG, é possível saber de onde veio cada trecho de informação usado para gerar uma resposta. Isso permite maior auditoria e rastreabilidade, o que é essencial em ambientes regulados (como jurídico ou saúde).

💰 5. Redução de custos com retreinamento
Ao invés de gastar tempo e dinheiro para retreinar um LLM com dados novos, você pode apenas atualizar a base de conhecimento consultada pelo RAG (por exemplo, adicionar novos arquivos à base vetorial).

# 🧰 Ferramentas Utilizadas

✅ Ollama

✅ Databricks

✅ Jupyter notabook

![image](https://github.com/user-attachments/assets/0931b8e6-a6bc-4e1d-b657-36d38f82b3fc)


# Vamos iniciar o projeto inicializando o Jupytrt Notebook:


📦 1. Instalação de Pacotes

Os pacotes necessários são instalados com pip, incluindo:

* LangChain: Framework para orquestração de LLMs.

* HuggingFace: Para embeddings.

* Milvus: Base vetorial para armazenamento e busca semântica.

* Databricks CLI e SQL Connector: Integração com o Databricks.

* BeautifulSoup (bs4): Para extrair textos de páginas HTML.



🔐 2. Configuração de Credenciais do Databricks

As credenciais do workspace do Databricks são definidas com variáveis de ambiente:

os.environ["DATABRICKS_HOST"] = "url_Databricks"
os.environ["DATABRICKS_TOKEN"] = "token"

Essas credenciais permitem que você use o LLM hospedado no Databricks.

![image](https://github.com/user-attachments/assets/b0548a6e-6ad3-45b7-9171-1f3e19099e06)
"Host"

![image](https://github.com/user-attachments/assets/0a5349e5-a4c8-426d-9a48-7256ddf99082)
"Tokens 90 diasa"


🌐 3. Extração de Dados da Web

Um artigo sobre o Microsoft Fabric é carregado diretamente da web:

dsa_loader = WebBaseLoader("https://blog.dsacademy.com.br/...")
documentos = dsa_loader.load()

![image](https://github.com/user-attachments/assets/7151f53c-49e4-48a2-9138-a67939378156)



🧩 4. Divisão do Texto em Chunks

O texto é dividido em partes menores (chunks) com RecursiveCharacterTextSplitter:

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
docs = text_splitter.split_documents(documentos)

Isso ajuda a manter o contexto relevante e reutilizável na recuperação vetorial.


🧠 5. Criação dos Embeddings

Os embeddings são vetores numéricos que representam semanticamente cada chunk de texto:

embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
Esse modelo é pequeno, rápido e adequado para tarefas de similaridade semântica.


🗃 6. Banco de Dados Vetorial (Milvus)

Os chunks com embeddings são armazenados em um banco vetorial:

dsa_vector_db = Milvus.from_documents(...)

Permite buscas por similaridade vetorial com base nas perguntas feitas.


🔎 7. Recuperação de Contexto

Com o .as_retriever(), criamos um mecanismo de busca semântica:

retriever = dsa_vector_db.as_retriever()

E o testamos com similarity_search("O Que é o Microsoft Fabric?").

![image](https://github.com/user-attachments/assets/8b45d6e2-fe80-4176-b69a-e99e770f7a6b)



🤖 8. Definição do LLM do Databricks

Aqui você define qual modelo LLM será usado para gerar as respostas:

llm = ChatDatabricks(endpoint = "databricks-dbrx-instruct", max_tokens = 200)

![image](https://github.com/user-attachments/assets/b3586dd6-2efc-4fed-9a39-3bdd0a283cf7)


🧾 9. Criação do Prompt Template

Define a estrutura da pergunta que será enviada ao modelo:

PROMPT_TEMPLATE = """..."""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
Esse prompt inclui um bloco <context> (com os dados recuperados) e <question> (a pergunta do usuário).


🔄 10. Definição da RAG Chain

Aqui é montada a "cadeia" de execução que conecta as etapas:

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

Explicando cada parte:

* retriever | format_docs: busca os documentos e formata em texto plano.

* RunnablePassthrough(): passa a pergunta diretamente.

* prompt: monta a entrada com base no contexto e na pergunta.

* llm: envia a entrada ao modelo do Databricks.

* StrOutputParser(): pega apenas a resposta em texto do modelo.


▶️ 11. Execução do Pipeline
A pipeline é invocada com .invoke():

resposta = rag_chain.invoke("Como Empresas Podem Utilizar o Microsoft Fabric?")

Isso retorna uma resposta gerada com base no texto recuperado da web + modelo LLM.

![image](https://github.com/user-attachments/assets/7605ecd9-8282-4698-ba8c-6cc0ecc30eca)


🧪 12. Resultados e Verificação

Você faz novas perguntas e o pipeline responde de forma precisa, contextualizada e confiável.

📌 Resumo Visual do Fluxo:

[Pergunta do Usuário]

        ↓
        
[Retriever busca os chunks relevantes no banco vetorial]

        ↓
        
[Prompt é preenchido com contexto + pergunta]

        ↓
        
[LLM Databricks gera uma resposta com base nisso]

        ↓
        
[Resposta Final]


# Abaixo está um diagrama simplificado do pipeline RAG que foi implementodo, agora vamos adaptá-lo a outros domínios (como jurídico ou financeiro).


📊 Diagrama do Pipeline RAG com LLM no Databricks

   +----------------------+
   
   | Pergunta do Usuário  |

   +----------------------+
   
            ↓
   +-------------------------------+
   
   | Recuperador (Retriever)      |
   
   | - Busca nos Embeddings       |
   
   | - Base Vetorial (Milvus)     |
   
   +-------------------------------+
   
            ↓
   +-------------------------------+
   
   | Contexto Recuperado           |
   
   +-------------------------------+
   
            ↓
   +-------------------------------+
   
   | Prompt Template               |
   
   | - Injeta pergunta e contexto |
   
   +-------------------------------+
   
            ↓
   +----------------------------------------+
   
   | Modelo LLM no Databricks     |
   
   | - endpoint: "databricks-dbrx-instruct" |
   
   +----------------------------------------+
   
            ↓
   +-------------------------------+
   
   | Resposta Gerada              |
  
   +-------------------------------+


🏛️ Como Adaptar o RAG para Domínios Específicos

🔹 1. Jurídico

Objetivo: Criar um assistente jurídico para consultas legais com base em leis atualizadas.

* Fonte de dados: sites de tribunais, portais jurídicos, legislação (como o Planalto.gov.br ou JusBrasil).

* Prompt: instruir o LLM a responder com base em dispositivos legais.

* Exemplo de pergunta: “Quais são os direitos do consumidor em caso de cancelamento de voo?”


🔹 2. Financeiro

Objetivo: Responder dúvidas sobre produtos bancários, investimentos ou normas da CVM/BC.

* Fonte de dados: sites do Banco Central, CVM, portais de notícias financeiras (Valor, InfoMoney).

* Prompt: garantir respostas com base em regulamentações e dados reais.

* Exemplo de pergunta: “Quais as regras para resgatar um CDB antes do vencimento?”


🔹 3. Saúde

Objetivo: Oferecer respostas sobre doenças, tratamentos, etc., com base em fontes médicas confiáveis.

* Fonte de dados: sites como Ministério da Saúde, OMS, PubMed.

* Prompt: alertar o modelo para não dar conselhos médicos finais (só informativos).

* Exemplo de pergunta: “Quais os sintomas da dengue?”


✅ Conclusão

A implementação de um pipeline RAG com LLMs no Databricks representa um passo significativo na construção de aplicações de IA Generativa mais robustas, precisas e seguras. Ao integrar recuperação de informações em tempo real com modelos de linguagem poderosos, superamos limitações tradicionais dos LLMs, como desatualização dos dados e risco de alucinações.

Durante este projeto, exploramos a estrutura técnica por trás do RAG, configuramos o ambiente com ferramentas como Milvus, LangChain, HuggingFace e Databricks, e demonstramos seu funcionamento com um exemplo prático baseado em dados da web. O resultado foi um sistema capaz de fornecer respostas contextualizadas e atualizadas, com rastreabilidade e flexibilidade para adaptação a diferentes domínios, como jurídico, financeiro e saúde.

Com essa base, o pipeline pode ser facilmente adaptado e expandido para outras áreas de negócio ou setores regulados, oferecendo soluções inteligentes com confiabilidade e escalabilidade. O futuro da IA Generativa está cada vez mais orientado a aplicações práticas com integração de dados dinâmicos — e o RAG é uma peça central nesse avanço.


