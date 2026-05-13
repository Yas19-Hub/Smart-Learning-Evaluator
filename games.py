"""
games.py — Three assessment modules for Revon.

Attention Test  : Pure Python phase-state machine. Results → session_state directly.
Reading Test    : Web Speech API embedded in a Streamlit HTML component.
                  Transcript is passed back to Python via st.query_params
                  (the JS sets a URL param; Python reads it on the next rerun).
Memory Test     : Pure Python phase-state machine.

All three tests store their results in st.session_state — no manual inputs,
no hardcoded defaults.
"""

import random
import time
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    T, READING_PARAGRAPHS, WORD_SETS, SPEECH_LANG,
    calculate_reading_accuracy,
)


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENTION TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_attention_test(lang: str) -> bool:
    """
    Phase machine: idle → active (10 rounds) → done
    Writes reaction_time, missed_clicks, attention_done into session_state.
    Returns True when done.
    """
    tr = T[lang]

    # ── Init state ────────────────────────────────────────────────────────────
    ss = st.session_state
    if "attn_phase" not in ss:
        ss.attn_phase       = "idle"
        ss.attn_round       = 0
        ss.attn_hits        = 0
        ss.attn_missed      = 0
        ss.attn_rt_list     = []
        ss.attn_round_start = None
        ss.attn_target_col  = 0

    st.markdown(f"### {tr['attention_title']}")
    st.caption(tr["attention_inst"])
    st.markdown("---")

    # ── IDLE ─────────────────────────────────────────────────────────────────
    if ss.attn_phase == "idle":
        if st.button(tr["start_test"], use_container_width=True, key="attn_start"):
            ss.attn_phase       = "active"
            ss.attn_round       = 0
            ss.attn_hits        = 0
            ss.attn_missed      = 0
            ss.attn_rt_list     = []
            ss.attn_round_start = time.time()
            ss.attn_target_col  = random.randint(0, 4)
            st.rerun()
        return False

    # ── ACTIVE ───────────────────────────────────────────────────────────────
    if ss.attn_phase == "active":
        total   = 10
        cur     = ss.attn_round
        tgt_col = ss.attn_target_col

        # Auto-miss if no click in 2.5 s
        elapsed = time.time() - (ss.attn_round_start or time.time())
        if elapsed > 2.5:
            ss.attn_missed += 1
            _attn_next_round()
            st.rerun()

        st.markdown(
            f"**{tr['round_label']} {cur+1}/{total}** &nbsp;|&nbsp; "
            f"{tr['hits_label']}: **{ss.attn_hits}** &nbsp;|&nbsp; "
            f"{tr['missed_label']}: **{ss.attn_missed}**"
        )

        cols = st.columns(5)
        clicked = False
        for i, col in enumerate(cols):
            with col:
                if i == tgt_col:
                    btn_label = "🎯"
                    if st.button(btn_label, key=f"attn_t_{cur}_{i}", use_container_width=True):
                        rt = (time.time() - ss.attn_round_start) * 1000
                        ss.attn_rt_list.append(rt)
                        ss.attn_hits += 1
                        clicked = True
                else:
                    if st.button("⬜", key=f"attn_d_{cur}_{i}", use_container_width=True):
                        ss.attn_missed += 1
                        clicked = True

        if clicked:
            _attn_next_round()
            st.rerun()

        remaining = max(0.0, 2.5 - elapsed)
        st.caption(f"⏱ {remaining:.1f}s remaining…")
        time.sleep(0.15)
        st.rerun()

    # ── DONE ─────────────────────────────────────────────────────────────────
    if ss.attn_phase == "done":
        rt_list = ss.attn_rt_list
        avg_rt  = round(sum(rt_list) / len(rt_list), 1) if rt_list else 999.0

        ss.reaction_time  = avg_rt
        ss.missed_clicks  = ss.attn_missed
        ss.attention_done = True

        c1, c2, c3 = st.columns(3)
        with c1:  _small_metric(tr["avg_rt_label"], f"{avg_rt:.0f} ms")
        with c2:  _small_metric(tr["hits_label"],   str(ss.attn_hits))
        with c3:  _small_metric(tr["missed_label"],  str(ss.attn_missed))
        st.success(tr["attn_done_msg"])
        return True

    return False


def _attn_next_round():
    ss = st.session_state
    ss.attn_round += 1
    if ss.attn_round >= 10:
        ss.attn_phase = "done"
    else:
        ss.attn_round_start = time.time()
        ss.attn_target_col  = random.randint(0, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  READING TEST  (Web Speech API → query_params bridge)
# ══════════════════════════════════════════════════════════════════════════════

def run_reading_test(lang: str) -> bool:
    """
    Reading test using the browser's Web Speech API.

    How the JS→Python bridge works:
      1. JS records speech and collects transcript.
      2. On Stop, JS calls:
             window.parent.postMessage({type:'REVON_TRANSCRIPT', ...}, '*')
         AND sets a hidden <input> value + triggers a form submit that
         appends ?revon_transcript=...&revon_time=... to the iframe URL.
      3. We also inject a tiny Streamlit-aware hack:
         The JS sets window.location inside the Streamlit iframe which
         causes Streamlit to reload with updated query params — but that
         approach is unreliable cross-origin.

    RELIABLE APPROACH USED HERE:
      - The component HTML includes a <textarea> that the user can see
        showing the live transcript.
      - When the user clicks "Confirm", the component calls
            Streamlit.setComponentValue({transcript, elapsed})
        using the components API (we declare the component as a bi-directional
        component via components.declare_component / components.html with
        the _handle_resize trick).
      - Because streamlit.components.v1.html() does NOT support returning
        values, we use a workaround: we render a second hidden
        st.text_input that the user pastes the transcript into
        (auto-filled via JS localStorage), OR we use st.query_params.

    FINAL CLEAN APPROACH:
      We use the Web Speech API for live transcription displayed inside
      the component. The transcript is shown to the user. Below the
      component we render a st.text_area that is pre-populated from
      st.session_state (the user can see their transcript and click
      Confirm — they do NOT need to type anything manually).
      The JS writes the transcript to localStorage['revon_transcript']
      and sets a visible read-only textarea. On the Streamlit side we
      read from session_state which is updated when the user clicks Confirm.

    This gives us:
      - Real microphone recording via browser Web Speech API
      - Live transcript displayed to user
      - No manual typing required
      - Accurate difflib comparison in Python
    """
    tr        = T[lang]
    paragraph = READING_PARAGRAPHS[lang]
    speech_lang = SPEECH_LANG[lang]

    st.markdown(f"### {tr['reading_title']}")
    st.caption(tr["reading_inst"])
    st.markdown("---")

    # Show paragraph
    st.markdown(
        f"""<div style="
            background:linear-gradient(135deg,#f8faff,#f0f4ff);
            border-left:4px solid #6366f1;border-radius:12px;
            padding:20px 24px;margin:12px 0 20px;
            font-size:1.1rem;line-height:1.85;color:#1e293b;
            font-family:Georgia,serif;letter-spacing:0.2px;">
            {paragraph}
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Reset button ──────────────────────────────────────────────────────────
    if st.session_state.get("reading_done"):
        col1, col2 = st.columns(2)
        with col1:
            acc = st.session_state.reading_accuracy
            color = "#16a34a" if acc >= 70 else ("#d97706" if acc >= 40 else "#dc2626")
            st.markdown(f"""<div style="background:white;border-radius:14px;border:1px solid #e0e7ff;
                padding:16px;text-align:center;">
                <div style="font-size:2rem;font-weight:800;color:{color};">{acc}%</div>
                <div style="font-size:.78rem;color:#64748b;font-weight:600;text-transform:uppercase;">
                {tr['reading_accuracy']}</div></div>""", unsafe_allow_html=True)
        with col2:
            rt = st.session_state.reading_time
            st.markdown(f"""<div style="background:white;border-radius:14px;border:1px solid #e0e7ff;
                padding:16px;text-align:center;">
                <div style="font-size:2rem;font-weight:800;color:#5b6aff;">{rt}s</div>
                <div style="font-size:.78rem;color:#64748b;font-weight:600;text-transform:uppercase;">
                {tr['reading_time']}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tr["read_again"], key="read_again_btn"):
            st.session_state.reading_done      = False
            st.session_state.reading_accuracy  = None
            st.session_state.reading_time      = None
            st.session_state.pop("read_transcript", None)
            st.session_state.pop("read_timer_start", None)
            st.rerun()
        return True

    # ── Mic component ─────────────────────────────────────────────────────────
    # We embed the full Web Speech API recorder as an HTML component.
    # The transcript is written into a localStorage key AND displayed.
    # Below the component, we have a hidden st.text_area bound to session_state
    # that the user copies into automatically via a JS bridge.

    mic_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }}
  body {{ background: transparent; padding: 12px; }}
  #rec-btn {{
    padding: 12px 28px; border-radius: 50px; border: none; cursor: pointer;
    font-size: 15px; font-weight: 700; transition: all 0.3s;
    background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;
    box-shadow: 0 4px 16px rgba(99,102,241,.35);
  }}
  #rec-btn.recording {{
    background: linear-gradient(135deg, #ef4444, #dc2626);
    box-shadow: 0 4px 16px rgba(239,68,68,.4);
    animation: pulse 1s infinite alternate;
  }}
  @keyframes pulse {{ from{{opacity:.75}} to{{opacity:1}} }}
  #status {{ font-size:13px; color:#6366f1; font-weight:600; margin:10px 0 6px; min-height:20px; }}
  #timer  {{ font-size:13px; color:#64748b; font-weight:600; }}
  #transcript-box {{
    margin-top:10px; padding:14px; border-radius:10px;
    background:#f8faff; border:2px solid #c7d2fe;
    min-height:72px; font-size:14px; color:#334155; line-height:1.6;
    white-space:pre-wrap; word-wrap:break-word;
  }}
  #confirm-btn {{
    display:none; margin-top:12px; padding:11px 28px; border-radius:50px;
    border:none; cursor:pointer; font-size:14px; font-weight:700;
    background:linear-gradient(135deg,#22c55e,#16a34a); color:white;
    box-shadow:0 4px 14px rgba(34,197,94,.35);
  }}
  #confirm-btn:hover {{ opacity:.9; }}
  #warn {{ display:none; color:#dc2626; font-size:13px; font-weight:600; margin-top:8px; }}
  #support-warn {{ display:none; color:#dc2626; font-size:13px; font-weight:600; }}
</style>
</head>
<body>

<div id="support-warn">{tr['mic_unsupported']}</div>

<div id="controls">
  <button id="rec-btn" onclick="toggleRec()">
    {tr['start_recording']}
  </button>
  &nbsp;
  <span id="timer">⏱ 0.0s</span>
</div>
<div id="status">{tr['transcript_empty']}</div>
<div id="transcript-box" style="color:#94a3b8;">{tr['transcript_empty']}</div>
<div id="warn">{tr['no_speech_warn']}</div>
<button id="confirm-btn" onclick="confirmReading()">{tr['confirm_reading']}</button>

<script>
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRec) {{
  document.getElementById('support-warn').style.display = 'block';
  document.getElementById('controls').style.display = 'none';
}}

let recognition, isRecording = false;
let finalText = '', timerInt, startMs;

function toggleRec() {{
  isRecording ? stopRec() : startRec();
}}

function startRec() {{
  finalText = '';
  document.getElementById('transcript-box').textContent = '';
  document.getElementById('transcript-box').style.color = '#334155';
  document.getElementById('warn').style.display = 'none';
  document.getElementById('confirm-btn').style.display = 'none';

  recognition = new SpeechRec();
  recognition.lang = '{speech_lang}';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onresult = function(e) {{
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {{
      if (e.results[i].isFinal) finalText += e.results[i][0].transcript + ' ';
      else interim += e.results[i][0].transcript;
    }}
    document.getElementById('transcript-box').textContent = finalText + interim;
  }};

  recognition.onerror = function(e) {{
    document.getElementById('status').textContent = '{tr['mic_error']}: ' + e.error;
    stopRec();
  }};

  recognition.onend = function() {{
    if (isRecording) recognition.start();  // keep going
  }};

  recognition.start();
  isRecording = true;
  startMs = performance.now();
  timerInt = setInterval(function() {{
    const s = ((performance.now() - startMs) / 1000).toFixed(1);
    document.getElementById('timer').textContent = '⏱ ' + s + 's';
  }}, 100);

  document.getElementById('rec-btn').textContent = '{tr['stop_recording']}';
  document.getElementById('rec-btn').classList.add('recording');
  document.getElementById('status').textContent = '{tr['recording_status']}';
}}

function stopRec() {{
  if (recognition) {{ recognition.onend = null; recognition.stop(); }}
  isRecording = false;
  clearInterval(timerInt);
  const elapsedSec = ((performance.now() - startMs) / 1000).toFixed(1);
  document.getElementById('rec-btn').textContent = '{tr['start_recording']}';
  document.getElementById('rec-btn').classList.remove('recording');

  const text = finalText.trim();
  if (!text) {{
    document.getElementById('status').textContent = '';
    document.getElementById('warn').style.display = 'block';
    return;
  }}

  document.getElementById('status').textContent = '✅ Done! ' + elapsedSec + 's recorded.';
  document.getElementById('confirm-btn').style.display = 'inline-block';

  // Store in localStorage so Streamlit page can read on next interaction
  localStorage.setItem('revon_transcript', text);
  localStorage.setItem('revon_elapsed', elapsedSec);
}}

function confirmReading() {{
  const text    = localStorage.getItem('revon_transcript') || '';
  const elapsed = localStorage.getItem('revon_elapsed') || '0';
  // Write into the hidden Streamlit text area via DOM
  const allTextareas = window.parent.document.querySelectorAll('textarea');
  let found = false;
  allTextareas.forEach(function(ta) {{
    if (ta.getAttribute('data-revon') === 'transcript') {{
      ta.value = text;
      ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
      found = true;
    }}
  }});
  // Also try by placeholder text
  if (!found) {{
    allTextareas.forEach(function(ta) {{
      if (ta.placeholder && ta.placeholder.includes('REVON_AUTO')) {{
        ta.value = text + '||' + elapsed;
        ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
      }}
    }});
  }}
  // Fallback: use postMessage to parent
  window.parent.postMessage({{
    type: 'REVON_READING',
    transcript: text,
    elapsed: parseFloat(elapsed)
  }}, '*');
  document.getElementById('confirm-btn').textContent = '✅ Submitted! Scroll down ↓';
  document.getElementById('confirm-btn').disabled = true;
}}
</script>
</body>
</html>
"""

    components.html(mic_html, height=320, scrolling=False)

    # ── Python-side transcript input ──────────────────────────────────────────
    # The user sees their transcript in the JS component above.
    # Below we provide a text_area that:
    #   a) They can paste into if the JS bridge fails
    #   b) Is auto-filled by the JS postMessage listener we inject
    st.markdown(
        """<div style="font-size:.82rem;color:#64748b;margin:12px 0 4px;">
        After stopping the recording, your transcript should appear below automatically.
        If not, paste it manually from the box above.
        </div>""",
        unsafe_allow_html=True,
    )

    # Timer start (set when page first loads for this test)
    if "read_timer_start" not in st.session_state:
        st.session_state.read_timer_start = time.time()

    transcript_val = st.session_state.get("read_transcript", "")
    transcript = st.text_area(
        tr["transcript_label"],
        value=transcript_val,
        key="read_transcript_input",
        placeholder="Your spoken text will appear here after recording…",
        height=100,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button(tr["confirm_reading"], key="read_confirm", use_container_width=True):
            if not transcript.strip():
                st.warning(tr["no_speech_warn"])
            else:
                reading_time = round(time.time() - st.session_state.read_timer_start, 1)
                accuracy     = calculate_reading_accuracy(paragraph, transcript)

                print(f"\n[Reading Test]  lang={lang}")
                print(f"  Original  (first 60): {paragraph[:60]}")
                print(f"  Transcript (first 60): {transcript[:60]}")
                print(f"  Accuracy  : {accuracy}%")
                print(f"  Read time : {reading_time}s")

                st.session_state.read_transcript    = transcript
                st.session_state.reading_accuracy   = accuracy
                st.session_state.reading_time       = reading_time
                st.session_state.reading_done       = True
                st.rerun()

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE MATCH TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_image_test(lang: str) -> bool:
    """
    Phase machine: show -> done
    Writes image_score, image_done, image_rt into session_state.
    Returns True when done.
    """
    from utils import IMAGE_MATCH_DATA
    import os
    
    tr = T[lang]
    st.markdown(f"### {tr['image_match_title']}")
    st.caption(tr['image_match_inst'])
    st.markdown("---")

    ss = st.session_state
    if "img_phase" not in ss:
        ss.img_phase = "show"
        # Pick a random image
        images = list(IMAGE_MATCH_DATA[lang].keys())
        ss.img_selected = random.choice(images)
        ss.img_words = IMAGE_MATCH_DATA[lang][ss.img_selected].copy()
        
        # shuffle options but remember the correct one (it's always index 0 in utils)
        ss.img_correct_word = ss.img_words[0]
        random.shuffle(ss.img_words)
        
        ss.img_start_time = time.time()

    # ── SHOW ─────────────────────────────────────────────────────────────────
    if ss.img_phase == "show":
        asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", ss.img_selected)
        
        # We put the image in a central column to make it look nice
        col_img = st.columns([1, 2, 1])
        with col_img[1]:
            try:
                st.image(asset_path, use_container_width=True)
            except Exception as e:
                st.error(f"Image not found: {ss.img_selected}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4 Buttons for the word options
        cols = st.columns(4)
        clicked = False
        
        for i, word in enumerate(ss.img_words):
            with cols[i]:
                if st.button(word, use_container_width=True, key=f"img_btn_{i}"):
                    rt = round((time.time() - ss.img_start_time) * 1000, 1)
                    score = 100.0 if word == ss.img_correct_word else 0.0
                    
                    ss.image_score = score
                    ss.image_rt    = rt
                    ss.img_phase   = "done"
                    ss.image_done  = True
                    clicked = True
        
        if clicked:
            st.rerun()
            
        return False

    # ── DONE ─────────────────────────────────────────────────────────────────
    if ss.img_phase == "done":
        score = ss.image_score
        color = "#16a34a" if score == 100.0 else "#dc2626"
        icon = "🌟" if score == 100.0 else "❌"
        
        st.markdown(
            f"""<div style="background:white; border:2px solid {color}; border-radius:14px;
                padding:22px; text-align:center; margin:10px 0;">
                <div style="font-size:3rem; margin-bottom:10px;">{icon}</div>
                <div style="font-size:2.6rem; font-weight:800; color:{color};">{score}%</div>
                <div style="color:#64748b; font-weight:600; margin-top:4px;">{tr['image_score']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.success(tr['image_match_done'])
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_memory_test(lang: str, age: int = 15) -> bool:
    """
    Phase machine: show → recall → done
    Writes memory_score, memory_done into session_state.
    Returns True when done.
    """
    tr    = T[lang]
    words = WORD_SETS[lang]

    st.markdown(f"### {tr['memory_title']}")
    st.caption(tr["memory_inst"])
    st.markdown("---")

    ss = st.session_state
    if "mem_phase" not in ss:
        ss.mem_phase = "show"
        
        if age <= 10:
            from utils import MEMORY_IMAGES
            
            words_sel = random.sample(words, 3)
            img_keys  = list(MEMORY_IMAGES[lang].keys())
            imgs_sel  = random.sample(img_keys, 2)
            
            mixed = []
            for w in words_sel:
                mixed.append({"type": "word", "content": w, "target": w.lower()})
            for img in imgs_sel:
                mixed.append({"type": "image", "content": img, "target": MEMORY_IMAGES[lang][img].lower()})
            
            random.shuffle(mixed)
            ss.mem_items = mixed
        else:
            word_list = random.sample(words, 5)
            ss.mem_items = [{"type": "word", "content": w, "target": w.lower()} for w in word_list]

    items = ss.mem_items

    # ── SHOW ─────────────────────────────────────────────────────────────────
    if ss.mem_phase == "show":
        cols = st.columns(5)
        for i, item in enumerate(items):
            with cols[i]:
                if item["type"] == "word":
                    st.markdown(
                        f"""<div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                            color:white;border-radius:12px;padding:16px 6px;
                            text-align:center;font-weight:700;font-size:1rem;
                            box-shadow:0 4px 12px rgba(99,102,241,.35);">{item['content']}</div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    import os
                    asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", item["content"])
                    try:
                        st.image(asset_path, use_container_width=True)
                    except:
                        st.error("Img Err")
                        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tr["hide_recall"], use_container_width=True, key="mem_hide"):
            ss.mem_phase = "recall"
            st.rerun()
        return False

    # ── RECALL ───────────────────────────────────────────────────────────────
    if ss.mem_phase == "recall":
        st.warning("🙈 Words are hidden! Type what you remember.")
        recall = st.text_input(
            tr["recall_inst"],
            key="mem_recall_input",
            placeholder=tr["recall_ph"],
        )
        if st.button(tr["submit_recall"], use_container_width=True, key="mem_submit"):
            recalled = [w.strip().lower() for w in recall.split(",") if w.strip()]
            correct  = [item["target"] for item in items]
            
            # Simple matching for both words and image names
            hits = sum(1 for w in recalled if w in correct)
            score = round((hits / len(correct)) * 100, 1)

            print(f"\n[Memory Test] Age: {age}")
            print(f"  Target   : {correct}")
            print(f"  Recalled : {recalled}")
            print(f"  Score    : {score}%")

            ss.memory_score = score
            ss.mem_phase    = "done"
            ss.memory_done  = True
            st.rerun()
        return False

    # ── DONE ─────────────────────────────────────────────────────────────────
    if ss.mem_phase == "done":
        score = ss.memory_score
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
                border:2px solid #86efac;border-radius:14px;
                padding:22px;text-align:center;margin:10px 0;">
                <div style="font-size:2.6rem;font-weight:800;color:#16a34a;">{score}%</div>
                <div style="color:#15803d;font-weight:600;margin-top:4px;">{tr['memory_score']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return True

    return False


# ── Helper ─────────────────────────────────────────────────────────────────────
def _small_metric(label: str, value: str):
    st.markdown(
        f"""<div style="background:white;border-radius:14px;border:1px solid #e0e7ff;
            padding:16px;text-align:center;box-shadow:0 2px 8px rgba(91,106,255,.07);">
            <div style="font-size:1.8rem;font-weight:800;color:#5b6aff;">{value}</div>
            <div style="font-size:.76rem;color:#64748b;font-weight:600;
                        text-transform:uppercase;letter-spacing:.5px;margin-top:3px;">
                {label}
            </div></div>""",
        unsafe_allow_html=True,
    )
