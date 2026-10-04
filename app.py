import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
SCHEMES = json.loads((BASE / 'data' / 'schemes.json').read_text(encoding='utf-8'))
st.set_page_config(page_title='CivicAid AI', page_icon='🤝', layout='wide')

st.markdown('''<style>
.hero {padding:1.4rem 1.6rem;border-radius:20px;background:linear-gradient(135deg,#172554,#0f766e);color:white;margin-bottom:1rem}
.hero h1 {margin:0;font-size:2.4rem}.hero p{margin:.45rem 0 0;opacity:.9;font-size:1.05rem}
.card {padding:1rem;border:1px solid #dbe4ea;border-radius:16px;background:#fff;margin-bottom:.8rem}
.badge {display:inline-block;padding:.22rem .55rem;border-radius:999px;background:#ecfeff;color:#155e75;font-size:.8rem}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><h1>🤝 CivicAid AI</h1><p>Find the support you may qualify for — understand why, prepare documents, and apply safely.</p></div>''', unsafe_allow_html=True)

st.sidebar.title('Your profile')
language = st.sidebar.selectbox('Language / भाषा', ['English', 'हिन्दी'])
age = st.sidebar.number_input('Age / आयु', 13, 100, 21)
state = st.sidebar.selectbox('State / राज्य', ['Madhya Pradesh','Delhi','Chhattisgarh','Nagaland','Other'])
income = st.sidebar.number_input('Annual family income (₹)', 0, 10000000, 150000, step=10000)
student = st.sidebar.checkbox('Student / विद्यार्थी', True)
disability = st.sidebar.checkbox('Person with disability / दिव्यांग', False)
minority = st.sidebar.checkbox('Minority community / अल्पसंख्यक', False)
rural = st.sidebar.checkbox('Rural resident / ग्रामीण', False)
education = st.sidebar.selectbox('Education', ['Class 10','Class 12','Diploma','B.E./B.Tech','M.E./M.Tech','PhD','Other'])
st.sidebar.markdown('---')
st.sidebar.caption('Prototype note: scheme data is a demo knowledge base. Always verify current eligibility and application instructions on the official source before applying.')

st.subheader('1 · Tell CivicAid what you need')
prompt = st.text_area('Describe your situation in your own words', placeholder='Example: I am an engineering student from Madhya Pradesh and my family income is low. I want scholarship or financial support.', height=110)

def profile_text():
    return ' '.join([f'age {age}', state.lower(), f'income {income}', education.lower(), 'student' if student else '', 'disability' if disability else '', 'minority' if minority else '', 'rural' if rural else '', prompt.lower()])

def rule_eligibility(scheme):
    checks=[]; criteria=scheme.get('criteria',{})
    if 'state' in criteria and criteria['state'] != 'Any' and state not in criteria['state']: checks.append('State/residence mismatch')
    if 'max_income' in criteria and income > criteria['max_income']: checks.append(f"Income above ₹{criteria['max_income']:,} limit")
    if criteria.get('student') and not student: checks.append('Student status required')
    if criteria.get('disability') and not disability: checks.append('Disability status required')
    if criteria.get('minority') and not minority: checks.append('Minority status required')
    allowed=criteria.get('education')
    if allowed and allowed != 'Any' and education not in allowed: checks.append('Education level may not match')
    return checks

docs_have={'Aadhaar card':False,'Income certificate':income<300000,'Domicile certificate':state!='Other','Bank passbook':False,'Marksheet':education!='Other','Disability certificate':disability,'Samagra ID':state=='Madhya Pradesh','Passport-size photo':False,'Ration card':False,'Student ID':student,'Acceptance letter':False,'Institution ID':student,'No Objection Certificate':False,'PAN card':False,'Educational certificates':education!='Other','Proposal/report':False}

corpus=[s['search_text'] for s in SCHEMES]
vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
X=vectorizer.fit_transform(corpus)
query_vector=vectorizer.transform([profile_text()])
semantic=cosine_similarity(query_vector,X).ravel()
results=[]
for i,scheme in enumerate(SCHEMES):
    mismatches=rule_eligibility(scheme); semantic_score=float(semantic[i])
    eligibility_score=1.0 if not mismatches else max(0.25,1-0.18*len(mismatches))
    missing=[d for d in scheme['documents'] if not docs_have.get(d,False)]
    readiness=max(0.0,1-len(missing)/max(1,len(scheme['documents'])))
    final=round(100*(0.55*eligibility_score+0.30*min(1,semantic_score*3.0)+0.15*readiness))
    results.append((final,semantic_score,mismatches,missing,scheme))
results.sort(key=lambda x:x[0],reverse=True)

tab1,tab2,tab3=st.tabs(['🎯 Find Support','📋 Document Readiness','🛡️ Application Safety'])
with tab1:
    st.caption('CivicAid combines profile rules + ML similarity matching. It does not make a legal or official eligibility decision.')
    if not prompt: st.info('Add a short description above for more personalized matching.')
    for final,sem,mismatches,missing,scheme in results:
        with st.container():
            st.markdown('<div class="card">',unsafe_allow_html=True)
            c1,c2=st.columns([4,1])
            with c1:
                st.markdown(f"### {scheme['name']}")
                st.markdown(f"**{scheme['category']}** · {scheme['state_scope']} · **Benefit:** {scheme['benefit']}")
                st.write(scheme['summary'])
                if mismatches: st.warning('Needs verification: '+' • '.join(mismatches))
                else: st.success('Profile appears aligned with the stored criteria.')
                if missing: st.write('**Missing / unconfirmed documents:** '+', '.join(missing))
                st.markdown(f"<span class='badge'>Why matched: {scheme['why']}</span>",unsafe_allow_html=True)
            with c2:
                st.metric('CivicAid score',f'{final}/100')
                st.link_button('Official source',scheme['source_url'])
            st.markdown('</div>',unsafe_allow_html=True)
with tab2:
    st.subheader('Your document readiness')
    all_docs=sorted(set(d for s in SCHEMES for d in s['documents']))
    rows=[{'Document':d,'Status':'✓ Available / likely' if docs_have.get(d,False) else '○ Not confirmed'} for d in all_docs]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.info('Demo feature: a production version could add consent-based document upload, OCR and automatic PII redaction.')
with tab3:
    st.subheader('Application Safety Check')
    st.write('CivicAid recommends applying through official source domains stored with each scheme.')
    official_domains={'myscheme.gov.in','search.myscheme.gov.in'}
    for scheme in SCHEMES:
        domain=re.sub(r'^https?://','',scheme['source_url']).split('/')[0]
        st.write(('🟢' if domain in official_domains else '🟠')+f" {scheme['name']} — {domain}")
    st.caption('Never pay a middleman or share OTP/PIN/passwords. Verify current instructions on the official portal.')
st.markdown('---')
st.caption('CivicAid AI • Student prototype • Responsible AI: recommendations are informational, not official eligibility determinations.')
