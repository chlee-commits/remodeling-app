import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# [상단 설정] 
st.set_page_config(page_title="리모델링 유사 프로젝트 분석 솔루션", layout="wide")

# [1. 유틸리티 함수]
def clean_value(val):
    if pd.isna(val) or val == '-': return 0.0
    val_str = str(val).replace(',', '').strip()
    if '%' in val_str:
        try: return float(val_str.replace('%', '')) / 100.0
        except: return 0.0
    try: return float(val_str)
    except: return 0.0

def calculate_relative_diff(target, db_val):
    """상대적 차이 계산"""
    denom = abs(target)
    if denom < 1e-9:
        return 0.0 if abs(db_val) < 1e-9 else 1.0
    return abs(target - db_val) / denom

# [2. 코사인 유사도 및 독립성 지수 기반 가중치 산출 엔진]
def derive_weights_logic(uploaded_file):
    df_s1 = pd.read_excel(uploaded_file, sheet_name="Sheet1", header=None)
    df_s1[1] = df_s1[1].ffill() 
    df_s1['wbs_key'] = df_s1.iloc[:, 3:10].apply(lambda x: x.astype(str).str.strip()).agg('-'.join, axis=1)
    all_wbs = df_s1['wbs_key'].unique()
    wbs_idx_map = {key: i for i, key in enumerate(all_wbs)}
    
    elements = ['세대수', '동수', '최고층수(지상)', '최고층수(지하)', '주차대수', '연면적', '대지면적', '건폐율', '용적률']
    n = len(elements)
    m = len(all_wbs)
    
    matrix = np.zeros((m, n))
    base_p = np.zeros(n) 

    for j, elem in enumerate(elements):
        escaped_elem = re.escape(elem)
        mask = df_s1[1].astype(str).str.contains(escaped_elem, na=False)
        relevant_data = df_s1[mask]
        for _, row in relevant_data.iterrows():
            w_idx = wbs_idx_map[row['wbs_key']]
            matrix[w_idx, j] += clean_value(row[10])
            prop = clean_value(row[12])
            if prop > 0:
                base_p[j] = max(base_p[j], prop)
    
    base_p[base_p == 0] = 0.01
    r_matrix = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            v_j, v_k = matrix[:, j], matrix[:, k]
            norm_jk = np.linalg.norm(v_j) * np.linalg.norm(v_k)
            if norm_jk == 0:
                r_matrix[j, k] = 1.0 if j == k else 0.0
            else:
                r_matrix[j, k] = np.dot(v_j, v_k) / norm_jk

    f_j = np.sum(1 - r_matrix, axis=1)
    c_j = base_p * f_j
    final_w = c_j / np.sum(c_j) if np.sum(c_j) != 0 else np.ones(n)/n
    
    return dict(zip(elements, final_w)), f_j, r_matrix, elements

# [3. 데이터 로드 함수]
def load_db(uploaded_file, items_list):
    def process_sheet(name):
        df = pd.read_excel(uploaded_file, sheet_name=name, header=None)
        proj_names = df.iloc[4, 3:].values
        raw_data = df.iloc[5:14, 3:]
        cleaned = raw_data.applymap(clean_value)
        cleaned.columns = [str(p).strip() for p in proj_names]
        cleaned.index = ['세대수', '동수', '최고층수(지상)', '최고층수(지하)', '주차대수', '연면적', '대지면적', '건폐율', '용적률']
        for idx in ['건폐율', '용적률']:
            cleaned.loc[idx] = cleaned.loc[idx].apply(lambda x: x/100.0 if x > 2.0 else x)
        return cleaned.T
    
    db_pre = process_sheet("리모델링전")
    db_post = process_sheet("리모델링후")
    db_chg = db_post[items_list] - db_pre[items_list]
    return db_pre[items_list], db_post[items_list], db_chg

# --- UI 메인 ---
st.title("🏗️ 리모델링 유사 프로젝트 분석 솔루션")
uploaded_file = st.file_uploader("📂 설계개요 정리파일(xlsx) 업로드", type=["xlsx"])

if uploaded_file:
    WEIGHTS_MAP, f_index, r_mat, item_names = derive_weights_logic(uploaded_file)
    ITEMS = list(WEIGHTS_MAP.keys())
    db_pre, db_post, db_chg = load_db(uploaded_file, ITEMS)

    st.subheader("📊 1. 항목별 산출 가중치")
    w_df = pd.DataFrame([{"항목": k, "독립성 지수": round(f_index[i], 4), "최종 가중치": round(WEIGHTS_MAP[k], 4)} for i, k in enumerate(ITEMS)]).set_index("항목")
    st.table(w_df.sort_values(by="최종 가중치", ascending=False).T)

    st.divider()
    st.subheader("📝 2. 신규 단지 정보 입력")
    first_pre, first_post = db_pre.iloc[0], db_post.iloc[0]
    tabs = st.tabs(["리모델링 전", "리모델링 후"])
    input_data = {}
    for t_idx, tab in enumerate(tabs):
        with tab:
            cols = st.columns(3)
            stage = "pre" if t_idx == 0 else "post"
            base_data = first_pre if t_idx == 0 else first_post
            for i, item in enumerate(ITEMS):
                key = f"{stage}_{item}_final"
                default_val = float(base_data[item]) * 1.1 
                if item in ['건폐율', '용적률']:
                    input_data[f"{stage}_{item}"] = cols[i%3].number_input(f"{item}(%)", value=default_val*100, key=key) / 100.0
                else:
                    input_data[f"{stage}_{item}"] = cols[i%3].number_input(f"{item}", value=default_val, key=key)
    for item in ITEMS:
        input_data[f"chg_{item}"] = input_data[f"post_{item}"] - input_data[f"pre_{item}"]

    st.divider()
    if st.button("🚀 유사도 정밀 분석 실행", use_container_width=True):
        full_calc_map = {item: {} for item in ITEMS}
        total_scores = pd.Series(0.0, index=db_pre.index)
        for item in ITEMS:
            w = WEIGHTS_MAP[item]
            for project in db_pre.index:
                d1 = calculate_relative_diff(input_data[f"pre_{item}"], db_pre.loc[project, item])
                d2 = calculate_relative_diff(input_data[f"post_{item}"], db_post.loc[project, item])
                d3 = calculate_relative_diff(input_data[f"chg_{item}"], db_chg.loc[project, item])
                avg_d = (d1 + d2 + d3) / 3.0
                score = w * avg_d
                full_calc_map[item][project] = [d1, d2, d3, avg_d, score]
                total_scores[project] += score
        res_df = total_scores.sort_values().to_frame(name="유사도 거리")
        res_df['유사도 점수(%)'] = (1 / (1 + res_df['유사도 거리'])) * 100
        st.session_state['res'] = res_df
        st.session_state['map'] = full_calc_map
        st.session_state['scores'] = total_scores

    if 'res' in st.session_state:
        res_df, calc_map, total_scores = st.session_state['res'], st.session_state['map'], st.session_state['scores']
        
        st.subheader("🔍 3. 단지별 상세 계산 과정")
        selected_project = st.selectbox("계산 근거를 확인할 단지 선택", res_df.index)
        breakdown = []
        for item in ITEMS:
            s = calc_map[item][selected_project]
            breakdown.append({
                "항목": item, "가중치(W)": round(WEIGHTS_MAP[item], 4),
                "리모델링 전": round(s[0], 6), "리모델링 후": round(s[1], 6), "증감": round(s[2], 6),
                "항목 기여도(%)": round((s[4] / total_scores[selected_project]) * 100, 2)
            })
        st.table(pd.DataFrame(breakdown).sort_values(by="항목 기여도(%)", ascending=False))
        st.info("💡 **항목 기여도**: 해당 항목의 차이가 전체 유사도 판정에 미친 영향력입니다.")

        st.divider()
        st.subheader("🏆 4. 최종 유사 프로젝트 매칭 결과")
        st.dataframe(res_df.style.highlight_min(subset=['유사도 거리'], color='lightgreen'), use_container_width=True)
        best_project = res_df.index[0]
        st.success(f"✅ 분석 결과, 입력 조건과 가장 유사한 단지는 **'{best_project}'** 입니다. (유사도: {res_df.iloc[0]['유사도 점수(%)']:.2f}%)")
        st.info("**💡 계산 원리 설명**\n- **리모델링 전/후**: 입력값 대비 DB 사례 단지의 수치적 차이 비율입니다.\n- **증감**: 리모델링 변화량(증가분)에 대한 차이 비율입니다.\n- **유사도 점수**: 모든 차이를 가중 합산하여 100% 환산한 지표입니다.")
else:
    st.warning("👈 왼쪽 상단에서 엑셀 파일을 업로드해 주세요.")

