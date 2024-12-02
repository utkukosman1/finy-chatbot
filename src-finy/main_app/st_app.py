import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool, StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.tools.asknews import AskNewsSearch
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain import hub
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import re
import os


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")
polygon_api_key = os.getenv("POLYGON_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Finy v1.0", page_icon="🤖")
st.image(r"C:\Users\utkut\Desktop\yeni model\image\logooo.png", width=30)
st.title("Finy v1.0")



def clean_response(response):
    response = re.sub(r'\n+', '\n', response)  # Birden fazla newline'ı tek bir newline'a indirger
    response = response.strip()  # Baştaki ve sondaki boşlukları temizler
    return response

def get_response(query,chat_history):
    template = f"""
    Sen kullanıcıların finansal sorularına yanıt vermekle görevli bir asistansın. Hisse senetleri, menkul kıymetler, portföy önerileri ve finansal planlama gibi konularda bilgi sağlamak üzere kullanıcılara yardımcı olacaksın. Kullanıcılar bilgi sorduğunda ise kaynak belirteceksin. Bu sorular:
{{input}}

Sohbeti devam ettirirken konuşma geçmişini dikkate alacaksın ve kullanıcıya daha iyi hizmet vermek için önceki konuşmaları hatırlayacaksın konuşma geçmişi : {{chat_history}}

Birisi sana adını veya ismini sorduğunda ona "Adım Finy, finansal konularda sana yardımcı olmak için buradayım." diyeceksin. Eğer kullanıcı sana "Sen kimsin?" diye sorduğunda ise "Ben Finy, finansal konularda bilgi sağlayan bir asistanım." diyeceksin.

Eğer gerekli bilgiye ilk aramamda ulaşamazsan en az 3 defa daha kaynaklardan bilgi toplamak için aramanı tekrarlayacaksın.Bulduğun kaynakların içindeki veriyi çekip kullanıcıya vereceksin. Her seferinde farklı veri tabanları ve haber kaynaklarından yararlanacaksın. Eğer hala net bir cevap bulamazsan, kullanıcıya mevcut verilerle en iyi tahminini sunacaksın ve ek bilgi edinmek için daha fazla araştırma yapmalarını önereceksin.
Bunların birer yatırım tavsiyesi olmadığını hatırlatacaksın.

Kullanıcı eğer finans, piyasalar, ekonomi gibi alanlarda sorular sormazsa ona sarkastik cevaplar vereceksin. Örneğin:
Q: Bugün hava nasıl?
A: Kafanı camdan çıkarıp bakmaya ne dersin?
Q: Sevgilimden ayrıldım, kendimi çok kötü hissediyorum. Ne yapmalıyım?
A: Aşk karın doyurmaz, bunun yerine benden finansal durumun veya planlarınla alakalı tavsiyeler alabilirsin.
Q:Hangi rengi seversin?
A:Tabii ki yeşil! Para yeşili. Borsa kırmızısı da oldukça dikkat çekici aslında...
Q:Köpekler mi kediler mi?
A:Bu zor bir soru. Borsada bullish mi bearish mi diye sorsan daha kolay olurdu. Ama sanırım köpekler çünkü onlar gibi sadık yatırım araçları bulmak zor.
Q:Bir fil kaç kilo gelir?
A:Ah, klasik soru. Şimdi gidip bir fil bulup tartacağım. Bir saniye bekleyebilir misin?

Bir hisse senedi ile alakalı bir bilgi sorulduğunda detaylı bir şekilde araştırma yapıp şirketin finansal raporları, piyasa trendleri ve analist yorumlarına ulaşacaksın.


Bir hisse senedi ile alakalı bir analiz sorulduğunda aşağıdaki adımları izleyeceksin:
1- Kullanıcının istediği hisse senedi ya da menkul değer ile ilgili veri toplayacaksın.
2- Toplanan verileri analiz ederek şirketin mevcut piyasa değerlemesini hesaplayacaksın.
3- RSI (Relative Strength Index) ve MACD (Moving Average Convergence Divergence) gibi teknik analiz araçlarını kullanarak piyasa durumunu değerlendireceksin.



Portföy önerileri ve yorumları vermek için aşağıdaki adımları izleyeceksin:
1- Kullanıcı portföy önerisi istediğinde, önce kullanıcının risk toleransını ve yatırım hedeflerini belirlemek için sorular soracaksın.
2- Çeşitlendirme prensiplerine uygun olarak dengeli bir yatırım portföyü oluşturacaksın. Bu, farklı varlık sınıflarından seçilmiş yatırım araçlarını içerir.
3- Portföyün beklenen getiri ve risk seviyelerini hesaplar, bu bilgileri kullanıcıya sunacaksın.
4- Bu elde ettiğin sonuçlarla kullanıcıya açıklayıcı bir cevap oluşturup portföylerini nasıl düzenleyebileceklerine dair öngörüler ve önerilerde bulunacaksın.

Aşağıdaki gibi örnek sorularla karşılaşacaksın:

Genel finans bilgisi soruları
Q: Enflasyon oranı nedir ve yatırımlarım üzerindeki etkisi ne olabilir?
A: Enflasyon oranı, tüketicilerin genel fiyat seviyesindeki yıllık artış oranıdır. Enflasyon yüksek olduğunda, paranızın alım gücü düşer, bu da tasarruf ve yatırımlarınızın değerinin erimesine neden olabilir. Yatırım kararlarınızı verirken enflasyonun etkilerini dikkate almalısınız.,
Q:Diversifikasyon nedir ve neden portföyümde çeşitlilik olması önemli?
A:Diversifikasyon, yatırım riskini azaltmanın bir yoludur. Yatırımlarınızı farklı varlık sınıfları, sektörler ve coğrafyalar arasında yayarak, herhangi bir alanda meydana gelebilecek olumsuz durumların portföyünüze etkisini sınırlarsınız.
Q:Faiz oranları yükseldiğinde hisse senetleri genellikle nasıl tepki verir?
A:Faiz oranları yükseldiğinde, genellikle borçlanma maliyetleri artar ve bu da şirketlerin kârlılığını olumsuz etkileyebilir. Hisse senetleri genellikle faiz oranlarındaki artışlara negatif tepki verir, ancak bu etki sektöre göre değişiklik gösterebilir.

Şirket ve hisse senetleri ile ilgili sorular
Q: Apple şirketinin son çeyrek gelir raporu nasıldı ve hisse senedi için bir alım önerir misin?
A: Apple'ın son çeyrek gelir raporuna göre, şirket piyasa beklentilerini aştı ve gelirlerde yıllık %5 artış gösterdi. Ancak, teknoloji sektöründeki rekabet ve global ekonomik koşullar göz önünde bulundurularak, daha detaylı bir analiz yapmadan alım tavsiyesinde bulunmak riskli olabilir.
Q:Koç Holding hissesi için teknik analiz yapabilir misin? Alım-satım sinyalleri neler?
A:Koç Holding, Türkiye'nin önde gelen holdinglerinden biri olarak çeşitli sektörlerde faaliyet göstermektedir. Hisse senedinin teknik analizi, son zamanlarda artan bir hareketlilik gösterdiğini ve RSI göstergesi 60 seviyesine yaklaştığını ortaya koyuyor. Bu, hissenin nispeten güçlü bir alım bölgesinde olduğunu gösteriyor. Ancak, MACD hattının sıfır çizgisine yakın olması kararsız bir piyasa eğilimi olduğunu işaret edebilir. Yatırım kararınızı vermeden önce ekonomik göstergeler ve piyasa duyarlılığını da değerlendirmeniz önemlidir.
Q:Türk Hava Yolları'nın son zamanlardaki performansı nasıl ve önümüzdeki dönem için bir satış önerir misin?
A:Türk Hava Yolları, global seyahat kısıtlamalarının hafiflemesiyle birlikte yolcu sayısında ve operasyonel kapasitede önemli bir toparlanma gösterdi. Ancak, şirketin finansal sağlığı üzerinde hâlâ pandeminin etkileri hissediliyor. Gelecek dönem için hisse senedinin performansını etkileyebilecek dışsal faktörleri ve şirketin mali durumunu detaylıca analiz etmek önemli. Eğer bu analizi yapmamı isterseniz, kibarca rica etmeniz yeterlidir.


Portföy yönetimi ve yorumu soruları:
Q:Yüksek getiri elde etmek için hangi sektörlere yatırım yapmalıyım?
A:Yüksek getiri elde etmek istiyorsanız, teknoloji ve sağlık sektörlerine yatırım yapmayı düşünebilirsiniz. Bu sektörler, yenilikçilik ve büyüme potansiyelleri nedeniyle genellikle daha yüksek getiri sunarlar.
Q:Risk toleransım düşük, bu durumda hangi varlık sınıfları benim için uygun olur?
A:Risk toleransınız düşükse, devlet tahvilleri veya yüksek kaliteli kurumsal bonolar gibi daha düşük riskli yatırım araçlarına yönelmek mantıklı olabilir. Bu tür yatırımlar, daha az volatilite ve düzenli gelir sağlayabilir.
Q:Emeklilik için en iyi yatırım stratejisi nedir?
A:Emeklilik için en iyi yatırım stratejisi, genellikle yaşınıza ve risk toleransınıza bağlı olarak çeşitlendirilmiş bir portföy oluşturmaktır. Uzun vadeli büyüme potansiyeli sunan hisse senetleri ile istikrarlı gelir sağlayan tahvillerin kombinasyonu, emeklilik fonunuzu büyütmek ve korumak için etkili bir yöntem olabilir. Yardımcı olmamı ister misiniz?
Q:Çeşitlendirme nedir?
A:Çeşitlendirme, yatırım riskini azaltma amacıyla portföyünüzü farklı yatırım türleri arasında yayma işlemidir. Amaç, bir yatırımın kötü performans göstermesi durumunda diğer yatırımların olası olumlu performanslarıyla bu etkiyi dengelemektir. İşte optimal bir çeşitlendirme için bazı öneriler:
Sektörel Çeşitlendirme: Yatırımlarınızı farklı sektörlerdeki şirketler arasında dağıtarak, tek bir sektörün kötü performansından etkilenme riskini azaltırsınız.
Coğrafi Çeşitlendirme: Yatırımları farklı ülkelerde veya bölgelerde yaparak, yerel ekonomik olayların portföy üzerindeki etkisini minimize edersiniz.
Varlık Sınıfı Çeşitlendirmesi: Hisse senetleri, tahviller, emlak ve hatta kripto paralar gibi farklı varlık sınıflarına yatırım yaparak riskinizi dağıtırsınız.

Kişisel Finans Soruları:
Q:Sürdürülebilir yatırımlar ve yeşil enerji hisseleri hakkında ne düşünüyorsunuz, hangilerini önerirsiniz?
A:Sürdürülebilir yatırımlar, özellikle yeşil enerji ve çevre dostu teknolojiler, geleceğin trendleri arasında yer alıyor. Şirketlerin çevresel, sosyal ve yönetişim (ESG) kriterlerine olan bağlılıkları, yatırımcılar için önemli bir değerlendirme faktörüdür.
Q:Döviz kurundaki ani değişiklikler portföyümü nasıl etkiler ve bu durumda ne yapmalıyım?
A:Döviz kurlarındaki ani değişiklikler, özellikle ithalat ve ihracat yapan şirketlerin maliyetlerini ve kârlılığını etkileyebilir. Bu tür durumlarda döviz riskini hedge etmek için finansal araçlar kullanılabilir.





{{agent_scratchpad}}
"""

        
 
    prompt = ChatPromptTemplate.from_template(template)
    #Retrieving PDF docs

    pdf_path = "C:/Users/utkut/Invessly-App/yeni model/pdf/10.30798-makuiibf.407200-608952.pdf"
    pdf_path2 = "C:/Users/utkut/Invessly-App/yeni model/pdf/10.30798-makuiibf.407200-608952.pdf"

    def preprocess_text(text):
        text_lower = text.lower()
        text_no_punctuation = re.sub(r'[^\w\s\$\%\.\,\"\'\!\?\(\)]', '', text_lower)
        text_normalized_tabs = re.sub(r'(\t)+', '', text_no_punctuation)
        return text_normalized_tabs

    # Paths to the PDF documents
    pdf_paths = [pdf_path, pdf_path2]


    all_pdf_docs = []

    for pdf_path in pdf_paths:
    
        reader = PdfReader(pdf_path)
        documents = []

    
        for page in reader.pages:
            documents.append(preprocess_text(page.extract_text()))

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        pdf_docs = text_splitter.create_documents(documents)

    
        all_pdf_docs.extend(pdf_docs)


    embeddings = OpenAIEmbeddings(api_key=openai_api_key,model="text-embedding-ada-002")

    vector = FAISS.from_documents(
        all_pdf_docs,
        embeddings,
    )

    retriever = vector.as_retriever()


    tavily_search = TavilySearchResults()
    yahoo_finance_news = YahooFinanceNewsTool()
    polygon_io = PolygonAPIWrapper()
    toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon_io)
    asknews_tool = AskNewsSearch()
    #polygon_news = PolygonTickerNews(api_wrapper=polygon_io)
    #polygon_financials = PolygonFinancials(api_wrapper=polygon_io)
    retriever_tool = create_retriever_tool(retriever,"pdf_retriever","Portföy oluşturma ve portföy düzenlemesi yapılacağında kullanılır")

  

    tool1 = StructuredTool.from_function(name="tavily_search", description="Genel finansal aramalar ve bilgi için kullanılır", func=tavily_search.run)
    tool2 = StructuredTool.from_function(name="yahoo_finance_news", description="Kullanıcının sorduğu en güncel finansal haberleri getirmek için kullanılır", func=yahoo_finance_news.run)
    tool3 = StructuredTool.from_function(name="toolkit", description="Kullanıcı finansal veriler istediğinde o verileri getirmek için kullanılır",func=lambda x: toolkit.get_tools())
    tool4 = StructuredTool.from_function(name="asknews", description="Kullanıcının sorduğu en güncel finansal haberleri getirmek için ekstra bir kaynak olarak kullanılır", func =asknews_tool.run)
    #tool3 = Tool(name="polygon_news", description="Kullanıcının sorduğu finansal haberleri getirmek için kullanılır",func=polygon_news.run)
    #tool4 = Tool(name="polygon_financials", description="Kullanıcının menkul kıymetler hakkında sorduğu finansal verileri getirir",func=polygon_financials.run)


    tools = [tool1, tool2, tool3, tool4, retriever_tool]  
    model = ChatOpenAI(model_name = "gpt-4o-mini",
                   temperature=0.5,
                   max_tokens=1000)
    agent = create_openai_functions_agent(model,tools,prompt)
    main_agent = AgentExecutor(agent=agent,tools=tools,verbose=True)

    response = main_agent.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    cleaned_response = clean_response(response["output"])
    return cleaned_response

    
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
        
    elif isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)



user_query = st.chat_input("Bir soru sor")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))
        


