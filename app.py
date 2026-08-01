"""
app.py — Revon: AI Learning Disability Detection System
========================================================
Pages:  Landing → Form → Attention → Reading → Memory → Results
• ALL UI text uses the T[] translation dictionary (English / Hindi).
• ALL feature values come from actual tests — no hardcoded defaults, no sliders.
• ML prediction uses sklearn LogisticRegression trained on synthetic data.
• Reading test uses Web Speech API mic → browser transcript → difflib accuracy.
"""
import time
import streamlit as st
# ── Page config (must be very first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Revon | Learning Disability Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)
from model import predict
from utils import (
    T, RECOMMENDATIONS,
    save_student_data, get_previous_record,
    compute_error_rate, compute_task_completion,
)
from games import run_attention_test, run_reading_test, run_memory_test, run_image_test
# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..800;1,9..40,300..800&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --pri:#5b6aff; --acc:#8b5cf6; --bg:#f7f8ff; --border:#e2e8ff;
  --muted:#64748b; --text:#1e2140; --radius:16px;
  --shadow:0 4px 24px rgba(91,106,255,.09);
}
html,body,[class*="css"] { font-family:'DM Sans',sans-serif!important; background:var(--bg)!important; color:var(--text)!important; }
.main .block-container    { padding:2rem 1.5rem 5rem!important; max-width:790px!important; }
header,#MainMenu,footer,.stDeployButton { display:none!important; }

/* Buttons */
.stButton>button {
  background:linear-gradient(135deg,#5b6aff,#8b5cf6)!important;
  color:white!important; border:none!important; border-radius:50px!important;
  padding:.7rem 2rem!important; font-size:.97rem!important; font-weight:700!important;
  font-family:'DM Sans',sans-serif!important;
  box-shadow:0 4px 16px rgba(91,106,255,.28)!important; transition:all .2s!important;
}
.stButton>button:hover { transform:translateY(-2px)!important; box-shadow:0 8px 24px rgba(91,106,255,.4)!important; }

/* Inputs */
.stTextInput>div>div>input,
.stTextArea textarea,
.stNumberInput>div>div>input {
  border-radius:12px!important; border:2px solid var(--border)!important;
  font-family:'DM Sans',sans-serif!important;
}
.stTextInput>div>div>input:focus,
.stTextArea textarea:focus { border-color:#5b6aff!important; box-shadow:0 0 0 3px rgba(91,106,255,.1)!important; }
/* Progress dots */
.prog-wrap { display:flex; gap:8px; justify-content:center; margin-bottom:1.2rem; align-items:center; }
.prog-dot  { width:36px; height:8px; border-radius:4px; background:#e0e7ff; transition:all .3s; }
.prog-dot.active { background:linear-gradient(90deg,#5b6aff,#8b5cf6); }
.prog-dot.done   { background:#86efac; }

/* Cards */
.rcard { background:white; border-radius:var(--radius); border:1px solid var(--border); padding:1.8rem; box-shadow:var(--shadow); margin-bottom:1.4rem; }

/* Metrics */
.met { background:white; border-radius:14px; border:1px solid var(--border); padding:1.2rem; text-align:center; box-shadow:0 2px 10px rgba(91,106,255,.06); }
.met-v { font-size:2rem; font-weight:800; line-height:1.1; }
.met-l { font-size:.76rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.5px; margin-top:3px; }

/* Insight rows */
.ins { display:flex; gap:10px; align-items:flex-start; padding:10px 14px; border-radius:10px; background:#f8faff; border:1px solid #e8edff; margin-bottom:7px; font-size:.91rem; line-height:1.55; color:#334155; }

/* Compare grid */
.cmp { display:grid; grid-template-columns:2fr 1fr 1fr; gap:8px; padding:9px 14px; border-radius:10px; background:#f8faff; border:1px solid #e8edff; margin-bottom:5px; font-size:.88rem; align-items:center; }

/* Section header */
.sh { font-size:1.1rem; font-weight:700; color:var(--text); margin:1.4rem 0 .7rem; display:flex; align-items:center; gap:8px; }
.sh::after { content:''; flex:1; height:1px; background:var(--border); margin-left:8px; }

/* Hero */
.hero { text-align:center; padding:3.2rem 2rem 2.2rem; background:linear-gradient(155deg,#eef2ff,#f5f0ff,#fdf4ff); border-radius:22px; border:1px solid #dde3ff; margin-bottom:2rem; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-40px; right:-40px; width:200px; height:200px; background:radial-gradient(circle,rgba(139,92,246,.12),transparent 70%); pointer-events:none; }
.brand   { font-family:'DM Serif Display',serif; font-size:4rem; letter-spacing:-2px; background:linear-gradient(135deg,#5b6aff,#8b5cf6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1; margin-bottom:.5rem; }
.tagline { font-size:1rem; color:var(--muted); font-weight:500; margin-bottom:1.4rem; }
.quote   { font-family:'DM Serif Display',serif; font-style:italic; font-size:1rem; color:#7c3aed; background:rgba(139,92,246,.07); border-radius:11px; padding:13px 20px; display:inline-block; max-width:480px; border:1px solid rgba(139,92,246,.15); margin-bottom:1.8rem; }

/* Debug box */
.dbg { background:#1e293b; border-radius:12px; padding:16px 20px; font-family:monospace; font-size:.83rem; color:#94a3b8; line-height:1.7; }
.dbg .k { color:#818cf8; } .dbg .v { color:#4ade80; } .dbg .h { color:#f472b6; font-weight:700; }

hr { border-color:var(--border)!important; margin:1.3rem 0!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    "page":             "landing",
    "language":         "English",
    "student":          {},
    # Attention
    "attention_done":   False,
    "reaction_time":    None,
    "missed_clicks":    None,
    # Reading
    "reading_done":     False,
    "reading_accuracy": None,
    "reading_time":     None,
    "read_transcript":  "",
    # Image Match
    "image_done":       False,
    "image_score":      None,
    "image_rt":         None,
    # Memory
    "memory_done":      False,
    "memory_score":     None,
    # Results
    "results":          None,
    "prev_record":      None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_all():
    saved_lang = st.session_state.get("language", "English")
    # Clear all test sub-keys
    for _key in list(st.session_state.keys()):
        del st.session_state[_key]
    for _k, _v in _DEFAULTS.items():
        st.session_state[_k] = _v
    st.session_state["language"] = saved_lang
    st.rerun()


def tr(key: str) -> str:
    return T[st.session_state.language].get(key, key)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def render_topbar():
    """Language selector always in the top-right."""
    _, col = st.columns([4, 1])
    with col:
        current = st.session_state.language
        lang = st.selectbox(
            tr("lang_label"),
            ["English", "Hindi"],
            index=0 if current == "English" else 1,
            key="lang_sel",
            label_visibility="collapsed",
        )
        if lang != current:
            st.session_state.language = lang
            # Reset tests when language changes (paragraphs differ)
            for k in ["reading_done", "reading_accuracy", "reading_time",
                      "read_transcript", "mem_phase", "mem_words",
                      "read_timer_start", "attn_phase"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
    # Label on the left
    with _:
        st.markdown(
            f"<span style='font-size:.84rem;color:var(--muted);font-weight:600;'>"
            f"{tr('lang_label')}</span>",
            unsafe_allow_html=True,
        )


def render_progress(step: int):
    names = tr("step_names")
    dots  = "".join(
        f'<div class="prog-dot {"active" if i==step else "done" if i<step else ""}" '
        f'title="{names[i]}"></div>'
        for i in range(5)
    )
    st.markdown(
        f"""<div style="text-align:center;margin-bottom:4px;">
              <span style="font-size:.82rem;color:var(--muted);font-weight:600;">
                {tr('step_label')} {step+1} {tr('of_label')} 5 — <b>{names[step]}</b>
              </span>
            </div>
            <div class="prog-wrap">{dots}</div>""",
        unsafe_allow_html=True,
    )


def sc(v, lo=40, hi=70):
    return "#16a34a" if v >= hi else ("#d97706" if v >= lo else "#dc2626")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_landing():
    render_topbar()
    st.markdown(f"""
    <div class="hero">
      <div class="brand">{tr('app_name')}</div>
      <div class="tagline">{tr('tagline')}</div>
      <div class="quote">{tr('quote')}</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(tr("get_started"), use_container_width=True):
            st.session_state.page = "form"
            st.rerun()

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:2rem;">
      <div class="rcard" style="padding:1.4rem 1rem;text-align:center;">
        <div style="font-size:2rem;margin-bottom:8px;">🎯</div>
        <div style="font-weight:700;font-size:.93rem;margin-bottom:4px;">Attention Test</div>
        <div style="color:var(--muted);font-size:.8rem;">Reaction time &amp; focus</div>
      </div>
      <div class="rcard" style="padding:1.4rem 1rem;text-align:center;">
        <div style="font-size:2rem;margin-bottom:8px;">🎙️</div>
        <div style="font-weight:700;font-size:.93rem;margin-bottom:4px;">Reading + Mic</div>
        <div style="color:var(--muted);font-size:.8rem;">Speech → accuracy score</div>
      </div>
      <div class="rcard" style="padding:1.4rem 1rem;text-align:center;">
        <div style="font-size:2rem;margin-bottom:8px;">🧠</div>
        <div style="font-weight:700;font-size:.93rem;margin-bottom:4px;">Memory Test</div>
        <div style="color:var(--muted);font-size:.8rem;">Short-term recall</div>
      </div>
    </div>""", unsafe_allow_html=True)


def page_form():
    render_topbar()
    render_progress(0)
    st.markdown(f'<div class="rcard"><h2 style="margin:0 0 1.4rem;color:var(--pri);">👤 {tr("student_info")}</h2>', unsafe_allow_html=True)

    name  = st.text_input(tr("full_name"), placeholder="—", value=st.session_state.student.get("name",""))
    c1,c2 = st.columns(2)
    with c1: age   = st.number_input(tr("age"),   min_value=4,  max_value=25, value=int(st.session_state.student.get("age",10)))
    with c2: grade = st.text_input(tr("grade"), placeholder="e.g. Grade 4", value=st.session_state.student.get("grade",""))
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(tr("next"), use_container_width=True):
        if not name.strip():
            st.error("Please enter a name." if st.session_state.language=="English" else "कृपया नाम दर्ज करें।")
        else:
            st.session_state.student     = {"name": name, "age": age, "grade": grade}
            st.session_state.prev_record = get_previous_record(name)
            st.session_state.page        = "attention"
            st.rerun()


def page_attention():
    render_topbar()
    render_progress(1)
    done = run_attention_test(st.session_state.language)
    if done:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tr("next"), use_container_width=True, key="attn_next"):
            st.session_state.page = "reading"
            st.rerun()


def page_reading():
    render_topbar()
    render_progress(2)
    run_reading_test(st.session_state.language)
    if st.session_state.reading_done:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tr("next"), use_container_width=True, key="read_next"):
            age = int(st.session_state.student.get("age", 10))
            if age <= 10:
                st.session_state.page = "image"
            else:
                st.session_state.page = "memory"
            st.rerun()


def page_image():
    render_topbar()
    render_progress(3)
    done = run_image_test(st.session_state.language)
    if done:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tr("next"), use_container_width=True, key="img_next"):
            st.session_state.page = "memory"
            st.rerun()


def page_memory():
    render_topbar()
    render_progress(4)
    age = int(st.session_state.student.get("age", 10))
    done = run_memory_test(st.session_state.language, age)
    if done:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tr("view_results"), use_container_width=True, key="mem_next"):
            st.session_state.page = "results"
            st.rerun()


def page_results():
    render_topbar()
    render_progress(5)

    # Guard
    missing = []
    if not st.session_state.attention_done or st.session_state.reaction_time is None:
        missing.append("Attention")
    if not st.session_state.reading_done or st.session_state.reading_accuracy is None:
        missing.append("Reading")
    age = int(st.session_state.student.get("age", 10))
    if age <= 10 and (not st.session_state.get("image_done") or st.session_state.get("image_score") is None):
        missing.append("Word Match")
    if not st.session_state.memory_done or st.session_state.memory_score is None:
        missing.append("Memory")
    if missing:
        st.error(f"Incomplete tests: {', '.join(missing)}. Please go back.")
        if st.button("← Go Back"):
            st.session_state.page = "attention" if "Attention" in missing else \
                                    "reading"   if "Reading"  in missing else "memory"
            st.rerun()
        return

    # ── Real feature values ───────────────────────────────────────────────────
    reading_time     = float(st.session_state.reading_time)
    reading_accuracy = float(st.session_state.reading_accuracy)
    reaction_time    = float(st.session_state.reaction_time)
    missed_clicks    = int(st.session_state.missed_clicks)
    memory_score     = float(st.session_state.memory_score)
    task_completion  = compute_task_completion(reading_time, reaction_time)
    error_rate       = compute_error_rate(missed_clicks)
    attention_score  = max(0.0, round(100 - missed_clicks*10 - max(0,(reaction_time-300)/10), 1))

    features = {
        "reading_time":     reading_time,
        "reading_accuracy": reading_accuracy,
        "reaction_time":    reaction_time,
        "missed_clicks":    missed_clicks,
        "memory_score":     memory_score,
        "task_completion":  task_completion,
        "error_rate":       error_rate,
    }

    # ── ML Prediction ─────────────────────────────────────────────────────────
    if st.session_state.results is None:
        with st.spinner("🤖 Running AI analysis…"):
            time.sleep(0.6)
            result = predict(features)
            st.session_state.results = result
            save_student_data({
                "name":       st.session_state.student.get("name",""),
                "age":        st.session_state.student.get("age",""),
                "grade":      st.session_state.student.get("grade",""),
                **features,
                "prediction": result["prediction"],
                "risk_level": result["risk_level"],
                "confidence": result["confidence"],
            })

    result = st.session_state.results
    lang   = st.session_state.language
    recs   = RECOMMENDATIONS[lang].get(result["prediction"], RECOMMENDATIONS["English"]["Normal"])
    
    student = st.session_state.student
    s_name  = student.get("name", "Unknown")
    s_age   = student.get("age", "-")
    s_grade = student.get("grade", "-")

    # ── Student Profile ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:white;border:1px solid #e0e7ff;border-radius:16px;
                padding:1.4rem;display:flex;justify-content:space-between;
                align-items:center;margin-bottom:1.4rem;box-shadow:0 2px 10px rgba(91,106,255,.05);">
        <div>
            <div style="font-size:1.4rem;font-weight:800;color:var(--text);">{s_name}</div>
            <div style="font-size:.85rem;color:var(--muted);font-weight:600;margin-top:4px;">
                {tr('age')}: {s_age} &nbsp;&nbsp;|&nbsp;&nbsp; {tr('grade')}: {s_grade}
            </div>
        </div>
        <div style="font-size:2.5rem;opacity:0.8;">🧑‍🎓</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Debug panel ───────────────────────────────────────────────────────────
    with st.expander(tr("debug_title"), expanded=False):
        rows = "".join(
            f"<div><span class='k'>{k:<24}</span> = <span class='v'>{v}</span></div>"
            for k,v in features.items()
        )
        rows += (f"<br><div><span class='h'>→ Prediction</span>: "
                 f"<span class='v'>{result['prediction']}</span></div>")
        rows += (f"<div><span class='h'>→ Confidence </span>: "
                 f"<span class='v'>{result['confidence']}%</span></div>")
        for lbl, prob in result["all_probs"].items():
            rows += f"<div><span class='k'>  {lbl:<22}</span> = <span class='v'>{prob}%</span></div>"
        st.markdown(f"<div class='dbg'>{rows}</div>", unsafe_allow_html=True)

    # ── Prediction banner ──────────────────────────────────────────────────────
    COLORS = {
        "Normal":              ("#f0fdf4","#16a34a","#dcfce7"),
        "Dyslexia Risk":       ("#fef2f2","#dc2626","#fee2e2"),
        "ADHD Risk":           ("#fff7ed","#c2410c","#ffedd5"),
        "Learning Difficulty": ("#fefce8","#92400e","#fef9c3"),
    }
    bg,fg,bdr = COLORS.get(result["prediction"],("#f0f4ff","#5b6aff","#e0e7ff"))
    st.markdown(f"""
    <div style="background:{bg};border:2px solid {bdr};border-radius:20px;
                padding:1.8rem;text-align:center;margin-bottom:1.4rem;">
      <div style="font-size:.82rem;font-weight:700;color:{fg};text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:7px;">{tr('prediction')}</div>
      <div style="font-size:2.1rem;font-weight:800;color:{fg};margin-bottom:8px;">
        {result['prediction']}
      </div>
      <div style="display:inline-block;background:white;border-radius:50px;padding:5px 18px;
                  font-size:.86rem;font-weight:700;color:{fg};border:2px solid {bdr};">
        {tr('risk_level')}: {result['risk_level']} &nbsp;|&nbsp;
        {tr('confidence')}: {result['confidence']}%
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Score summary ──────────────────────────────────────────────────────────
    st.markdown('<div class="sh">📊 Score Summary</div>', unsafe_allow_html=True)
    
    image_score = st.session_state.get("image_score")
    if image_score is not None:
        c1,c2,c3,c4_img = st.columns(4)
    else:
        c1,c2,c3 = st.columns(3)
        c4_img = None
        
    with c1: st.markdown(f"""<div class="met"><div class="met-v" style="color:{sc(reading_accuracy)};">{reading_accuracy}%</div><div class="met-l">{tr('reading_accuracy')}</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="met"><div class="met-v" style="color:{sc(attention_score)};">{attention_score:.0f}</div><div class="met-l">{tr('attention_score')}</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="met"><div class="met-v" style="color:{sc(memory_score)};">{memory_score}%</div><div class="met-l">{tr('memory_score')}</div></div>""", unsafe_allow_html=True)
    if c4_img is not None:
        with c4_img: st.markdown(f"""<div class="met"><div class="met-v" style="color:{sc(image_score)};">{image_score:.0f}%</div><div class="met-l">{tr('image_score')}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4,c5,c6 = st.columns(3)
    mc_color = "#dc2626" if missed_clicks>5 else ("#d97706" if missed_clicks>2 else "#16a34a")
    with c4: st.markdown(f"""<div class="met"><div class="met-v" style="color:var(--pri);font-size:1.6rem;">{reaction_time:.0f}ms</div><div class="met-l">{tr('reaction_time')}</div></div>""", unsafe_allow_html=True)
    with c5: st.markdown(f"""<div class="met"><div class="met-v" style="color:{mc_color};">{missed_clicks}</div><div class="met-l">{tr('missed_clicks')}</div></div>""", unsafe_allow_html=True)
    with c6: st.markdown(f"""<div class="met"><div class="met-v" style="color:var(--pri);font-size:1.6rem;">{reading_time:.1f}s</div><div class="met-l">{tr('reading_time')}</div></div>""", unsafe_allow_html=True)

    # ── Probability breakdown ──────────────────────────────────────────────────
    st.markdown(f'<div class="sh">📈 {tr("prob_breakdown")}</div>', unsafe_allow_html=True)
    for lbl, prob in result["all_probs"].items():
        bar_col = fg if lbl == result["prediction"] else "#c7d2fe"
        st.markdown(f"""
        <div style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;font-size:.85rem;
                      font-weight:{'700' if lbl==result['prediction'] else '400'};margin-bottom:3px;">
            <span>{lbl}</span><span>{prob}%</span>
          </div>
          <div style="height:8px;background:#e0e7ff;border-radius:4px;overflow:hidden;">
            <div style="width:{prob}%;height:100%;background:{bar_col};border-radius:4px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Previous session comparison ────────────────────────────────────────────
    prev = st.session_state.get("prev_record")
    if prev:
        st.markdown(f'<div class="sh">📊 {tr("prev_session")} vs {tr("curr_session")}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="cmp" style="background:#e0e7ff;font-weight:700;font-size:.8rem;text-transform:uppercase;">
            <div>Metric</div><div>{tr('prev_session')}</div><div>{tr('curr_session')}</div></div>""", unsafe_allow_html=True)

        def cmp_row(label, p_key, curr):
            try:
                p = float(prev.get(p_key, 0))
                c = float(curr)
                arrow = "↑" if c>p else ("↓" if c<p else "→")
                col   = "#16a34a" if c>=p else "#dc2626"
                return (f'<div class="cmp"><div>{label}</div><div>{p}</div>'
                        f'<div style="color:{col};font-weight:700;">{c} {arrow}</div></div>')
            except: return ""

        st.markdown(
            cmp_row(tr("reading_accuracy"), "reading_accuracy", reading_accuracy) +
            cmp_row(tr("memory_score"),     "memory_score",     memory_score) +
            cmp_row(tr("reaction_time"),    "reaction_time",    reaction_time),
            unsafe_allow_html=True,
        )
        try:
            if reading_accuracy > float(prev.get("reading_accuracy",0)): st.success(tr("impr_read_up"))
            else: st.warning(tr("impr_read_dn"))
        except: pass
        try:
            if reaction_time < float(prev.get("reaction_time",9999)): st.success(tr("impr_attn_up"))
            else: st.info(tr("impr_attn_dn"))
        except: pass

    # ── Insights / Recommendations / Next Steps ───────────────────────────────
    for sec_key, icon, item_key in [
        ("insights","💡","insights"),
        ("recommendations","✅","recommendations"),
        ("next_steps","🚀","next_steps"),
    ]:
        st.markdown(f'<div class="sh">{tr(sec_key)}</div>', unsafe_allow_html=True)
        for item in recs[item_key]:
            st.markdown(f'<div class="ins"><span>{icon}</span><span>{item}</span></div>', unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#f8faff;border:1px solid #e0e7ff;border-radius:12px;
                padding:13px 17px;font-size:.8rem;color:#64748b;margin-top:1.4rem;
                text-align:center;line-height:1.6;">
      {tr('disclaimer')}
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button(tr("start_over"), use_container_width=True):
            reset_all()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════

_page = st.session_state.page
if   _page == "landing":   page_landing()
elif _page == "form":      page_form()
elif _page == "attention": page_attention()
elif _page == "reading":   page_reading()
elif _page == "image":     page_image()
elif _page == "memory":    page_memory()
elif _page == "results":   page_results()
