import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone
import hashlib
import jwt
import uuid
import os
from dotenv import load_dotenv
from openai import OpenAI
import pytz
from streamlit_autorefresh import st_autorefresh
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# OpenAI API client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Upbit API credentials
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
UPBIT_SERVER_URL = 'https://api.upbit.com'
api_key = os.getenv('NEWS_API_KEY')

# Generate JWT token
def generate_jwt_token():
    payload = {
        'access_key': UPBIT_ACCESS_KEY,
        'nonce': str(uuid.uuid4()),
    }
    jwt_token = jwt.encode(payload, UPBIT_SECRET_KEY)
    return jwt_token

# 1. Get account balance
def get_account_balance():
    headers = {
        'Authorization': f'Bearer {generate_jwt_token()}',
    }
    response = requests.get(UPBIT_SERVER_URL + '/v1/accounts', headers=headers)
    
    if response.status_code != 200:
        st.error(f"업비트 자산 조회 오류: {response.status_code}")
        return pd.DataFrame()

    balances = response.json()
    krw_balance = 0
    btc_balance = 0
    for balance in balances:
        if balance['currency'] == 'KRW':
            krw_balance = float(balance['balance'])
        elif balance['currency'] == 'BTC':
            btc_balance = float(balance['balance'])

    st.write("KRW 잔고:", krw_balance)
    st.write("BTC 잔고:", btc_balance)

    return pd.DataFrame({
        'Asset': ['KRW', 'BTC'],
        'Balance': [krw_balance, btc_balance]
    })

# 2. Get BTC/KRW data with adjustable interval (e.g., minutes/1 or hours/1)
def get_btc_data(interval='minutes/60'):  # 1시간 봉으로 조정 가능
    url = f'{UPBIT_SERVER_URL}/v1/candles/{interval}'
    params = {
        'market': 'KRW-BTC',
        'count': 180  # 요청 데이터 갯수
    }
    headers = {
        'Authorization': f'Bearer {generate_jwt_token()}',
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        st.error(f"업비트 데이터 가져오기 오류: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['candle_date_time_utc']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    df['close'] = df['trade_price'].astype(float)
    df['volume'] = df['candle_acc_trade_volume'].astype(float)
    return df[['timestamp', 'close', 'volume']]

# 3. Calculate additional indicators (MACD, Bollinger Bands, etc.)
def calculate_indicators(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['HULL'] = ta.hma(df['close'], length=20)

    # MACD는 기본적으로 3개의 컬럼을 반환: 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']  # MACD 값
    df['MACD_signal'] = macd['MACDs_12_26_9']  # MACD 신호선

    # Bollinger Bands는 5개의 값을 반환할 수 있으므로 필요한 것만 선택
    bbands = ta.bbands(df['close'], length=20)
    df['Bollinger_lower'] = bbands['BBL_20_2.0']  # Lower Band
    df['Bollinger_middle'] = bbands['BBM_20_2.0']  # Middle Band
    df['Bollinger_upper'] = bbands['BBU_20_2.0']  # Upper Band

    df = df.dropna(subset=['RSI', 'HULL'])  # NaN 값이 있는 행 제거
    return df

# 뉴스 API를 사용하여 최신 비트코인 관련 뉴스 헤드라인 가져오기
def get_latest_news_headlines():
     # 여기에 NewsAPI API 키를 넣으세요
    url = "https://newsapi.org/v2/everything"
    
    # 관심 있는 주제 (예: Bitcoin, Cryptocurrency)
    query_params = {
        'q': 'bitcoin',
        'language': 'en',  # 뉴스 언어 설정
        'sortBy': 'publishedAt',  # 최신 뉴스부터 정렬
        'apiKey': api_key
    }

    response = requests.get(url, params=query_params)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        # 헤드라인 리스트로 추출
        headlines = [article['title'] for article in articles[:5]]  # 상위 5개 헤드라인만 추출
        return headlines
    else:
        return ["Failed to retrieve news"]

# 매매 기록 저장 (간단한 예시로, 실제 매매 기록을 유지하는 함수)
def save_trade_record(trade_records, decision, current_price):
    new_record = pd.DataFrame({
        'Timestamp': [datetime.now()],
        'Action': [decision],
        'Price': [current_price]
    })
    return pd.concat([trade_records, new_record], ignore_index=True)

# OpenAI를 통해 과거 매매 기록과 함께 결정
def openai_advanced_decision_with_trade_history(df, trade_records, news_headlines=None):
    latest_data = df.iloc[-1]
    
    # 과거 50개의 데이터를 요약
    past_data_summary = df[['timestamp', 'close', 'RSI', 'MACD', 'Bollinger_upper', 'Bollinger_lower']].tail(50)

    # 매매 기록을 포함하여 OpenAI에게 학습시킴
    trade_summary = trade_records.tail(10).to_string() if not trade_records.empty else "No trades yet."

    # 프롬프트에 과거 매매 기록과 지표 데이터, 뉴스 헤드라인을 포함
    prompt = f"""
    Based on the past 50 data points and indicators:
    RSI: {latest_data['RSI']},
    MACD: {latest_data['MACD']},
    Bollinger Upper: {latest_data['Bollinger_upper']},
    Bollinger Lower: {latest_data['Bollinger_lower']},
    Recent trades: {trade_summary},
    """
    
    if news_headlines:
        prompt += f"Also consider the following recent news headlines: {news_headlines}\n"
    
    # 간결한 매매 신호 요청
    prompt += "Give only one word: BUY, SELL, or HOLD."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a cryptocurrency trading expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3,  # 최소 토큰 사용
        temperature=0.0  # 결과의 일관성 유지
    )
    
    return response.choices[0].message.content.strip()

# 5. Execute buy order
def execute_buy_order():
    account_balance = get_account_balance()
    krw_balance = account_balance[account_balance['Asset'] == 'KRW']['Balance'].values[0]
    current_btc_price = get_btc_data().iloc[-1]['close']
    
    amount_in_krw = krw_balance * 0.99  # 수수료 고려
    btc_to_buy = amount_in_krw / current_btc_price

    query = {
        'market': 'KRW-BTC',
        'side': 'bid',
        'price': str(amount_in_krw),
        'ord_type': 'price',
    }

    query_string = '&'.join([f'{key}={value}' for key, value in query.items()])
    query_hash = hashlib.sha512(query_string.encode()).hexdigest()

    payload = {
        'access_key': UPBIT_ACCESS_KEY,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, UPBIT_SECRET_KEY)
    headers = {'Authorization': f'Bearer {jwt_token}'}

    response = requests.post(UPBIT_SERVER_URL + '/v1/orders', params=query, headers=headers)

    if response.status_code == 201:
        st.write("매수 주문이 성공적으로 접수되었습니다.")
    else:
        st.write(f"매수 주문 실패: {response.status_code}, {response.text}")

# 6. Execute sell order
def execute_sell_order():
    account_balance = get_account_balance()
    btc_balance = account_balance[account_balance['Asset'] == 'BTC']['Balance'].values[0]

    if btc_balance > 0.0001:
        query = {
            'market': 'KRW-BTC',
            'side': 'ask',
            'volume': str(btc_balance),
            'ord_type': 'market',
        }

        query_string = '&'.join([f'{key}={value}' for key, value in query.items()])
        query_hash = hashlib.sha512(query_string.encode()).hexdigest()

        payload = {
            'access_key': UPBIT_ACCESS_KEY,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }

        jwt_token = jwt.encode(payload, UPBIT_SECRET_KEY)
        headers = {'Authorization': f'Bearer {jwt_token}'}

        response = requests.post(UPBIT_SERVER_URL + '/v1/orders', params=query, headers=headers)

        if response.status_code == 201:
            st.write("매도 주문이 성공적으로 접수되었습니다.")
        else:
            st.write(f"매도 주문 실패: {response.status_code}, {response.text}")
    else:
        st.write("매도를 위한 BTC 잔고가 부족합니다.")

# 7. Plot indicators with additional features
def plot_indicators(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("BTC 가격", "RSI"))

    # BTC 가격 라인
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='BTC 가격'), row=1, col=1)
    
    # RSI 라인
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)

    # 차트 레이아웃 크기 조정
    fig.update_layout(height=400, width=600)  # 높이와 너비 설정
    return fig

# Streamlit UI and control flow
st.title("BTC/KRW 자동매매 시스템")

# 초기 설정 (세션 상태)
if 'trade_records' not in st.session_state:
    st.session_state.trade_records = pd.DataFrame(columns=['Timestamp', 'Action', 'Price'])  # 매매 기록

# 최신 비트코인 관련 뉴스 헤드라인 표시
news_headlines = get_latest_news_headlines()
st.subheader("최신 비트코인 관련 뉴스")
for headline in news_headlines:
    st.write(f"- {headline}")

# Load data and calculate indicators
df = get_btc_data()  
df = calculate_indicators(df)

if not df.empty:
    # 차트 표시
    st.subheader("BTC/KRW 데이터와 지표")
    st.plotly_chart(plot_indicators(df))

    # 매매 기록을 활용한 OpenAI 결정
    ai_decision = openai_advanced_decision_with_trade_history(df, st.session_state.trade_records, news_headlines)
    
    st.subheader("OpenAI 매매 신호")
    st.write(f"AI 매매 신호: {ai_decision}")
    
    # 자동 매매 실행
    if ai_decision == "BUY":
        st.write("BTC 매수 실행 중...")
        execute_buy_order()
    elif ai_decision == "SELL":
        st.write("BTC 매도 실행 중...")
        execute_sell_order()
    else:
        st.write("BTC 보유 중...")
    
    # 매매 기록 업데이트
    st.session_state.trade_records = save_trade_record(st.session_state.trade_records, ai_decision, df.iloc[-1]['close'])

    # 매매 기록을 테이블로 출력
    st.subheader("매매 기록")
    st.dataframe(st.session_state.trade_records)
else:
    st.error("BTC 데이터를 가져올 수 없습니다.")

# Auto-refresh every minute
st_autorefresh(interval=3600 * 1000, limit=None, key="auto_refresh")
