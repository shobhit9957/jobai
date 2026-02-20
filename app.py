import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
import streamlit as st
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

APP_NAME = "Job AI"
HEADLINE = "One Place for all your job difficulties. Get a Job today with Job AI"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")


@dataclass
class UserProfile:
    id: str
    name: str = ""
    email: str = ""
    location: str = ""
    skills: str = ""
    preferred_location: str = ""
    projects: str = ""
    education: str = ""
    certifications: str = ""
    years_experience: str = ""
    target_role: str = ""
    linkedin_url: str = ""
    profile_pic: str = ""


def save_profile(profile: UserProfile) -> Path:
    output = DATA_DIR / f"profile_{profile.id}.txt"
    output.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    return output


def load_profile(user_id: str) -> UserProfile | None:
    file_path = DATA_DIR / f"profile_{user_id}.txt"
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return UserProfile(**payload)


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def ollama_chat(prompt: str, system: str | None = None) -> str:
    body = {
        "model": OLLAMA_MODEL,
        "messages": [],
        "stream": False,
    }
    if system:
        body["messages"].append({"role": "system", "content": system})
    body["messages"].append({"role": "user", "content": prompt})

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def linkedin_auth_url() -> str:
    if not LINKEDIN_CLIENT_ID or not LINKEDIN_REDIRECT_URI:
        return ""
    params = {
        "response_type": "code",
        "client_id": LINKEDIN_CLIENT_ID,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "scope": "openid profile email",
    }
    return f"https://www.linkedin.com/oauth/v2/authorization?{urlencode(params)}"


def google_auth_url() -> str:
    if not GOOGLE_CLIENT_ID or not GOOGLE_REDIRECT_URI:
        return ""
    params = {
        "response_type": "code",
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


def import_linkedin_profile_from_file(uploaded_file: Any) -> dict[str, Any]:
    raw = uploaded_file.read().decode("utf-8")
    data = json.loads(raw)
    return {
        "name": data.get("name", ""),
        "email": data.get("email", ""),
        "location": data.get("location", ""),
        "skills": ", ".join(data.get("skills", [])) if isinstance(data.get("skills"), list) else data.get("skills", ""),
        "projects": "\n".join(data.get("projects", [])) if isinstance(data.get("projects"), list) else data.get("projects", ""),
        "education": "\n".join(data.get("education", [])) if isinstance(data.get("education"), list) else data.get("education", ""),
        "certifications": "\n".join(data.get("certifications", [])) if isinstance(data.get("certifications"), list) else data.get("certifications", ""),
        "linkedin_url": data.get("linkedin_url", ""),
        "profile_pic": data.get("profile_pic", ""),
    }


def discover_jobs(role: str, geography: str, remote_ok: bool, count: int) -> list[dict[str, str]]:
    remote_hint = "remote" if remote_ok else geography
    prompt = f"""
Create a realistic list of {count} open jobs for role '{role}' in '{geography}' (or {remote_hint}).
Return STRICT JSON array with objects containing keys:
- title
- company
- location
- job_url
- recruiter_name
- recruiter_profile
- summary
"""
    output = ollama_chat(prompt, system="Return only valid JSON.")
    parsed = extract_json_payload(output)
    if isinstance(parsed, list):
        return parsed[:count]
    return []


def build_resume(profile: UserProfile, jobs: list[dict[str, str]]) -> str:
    job_summaries = "\n\n".join(
        [f"{j.get('title')} at {j.get('company')} ({j.get('location')})\n{j.get('summary')}" for j in jobs]
    )
    prompt = f"""
Create an ATS optimized resume for this candidate profile:
{json.dumps(asdict(profile), indent=2)}

Target jobs:
{job_summaries}

Requirements:
- include strong action verbs and quantifiable impacts
- include SEO and ATS keywords from target roles
- concise and professional
- output as markdown
"""
    return ollama_chat(prompt)


def recruiter_message(profile: UserProfile, job: dict[str, str]) -> str:
    prompt = f"""
Write a concise LinkedIn message to recruiter {job.get('recruiter_name')}.
Candidate name: {profile.name}
Applied role: {job.get('title')} at {job.get('company')}
Tone: respectful, personalized, confident, under 500 characters.
Mention one relevant strength from skills: {profile.skills}
"""
    return ollama_chat(prompt)


def referral_message(profile: UserProfile, role: str) -> str:
    prompt = f"""
Write a referral request message for LinkedIn.
Candidate: {profile.name}
Role: {role}
Skills: {profile.skills}
Keep it short, human, and professional.
"""
    return ollama_chat(prompt)


def linkedin_agent_run(
    jobs: list[dict[str, str]], recruiter_texts: dict[str, str], referral_note: str, dry_run: bool
) -> dict[str, Any]:
    cookie = os.getenv("LINKEDIN_SESSION_COOKIE", "")
    csrf = os.getenv("LINKEDIN_CSRF_TOKEN", "")
    if not cookie:
        return {"ok": False, "error": "Missing LINKEDIN_SESSION_COOKIE in environment."}

    report = {"applied": [], "messaged": [], "referrals": [], "errors": [], "dry_run": dry_run}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        context.add_cookies(
            [
                {
                    "name": "li_at",
                    "value": cookie,
                    "domain": ".linkedin.com",
                    "path": "/",
                    "httpOnly": True,
                    "secure": True,
                }
            ]
        )
        if csrf:
            context.add_cookies(
                [
                    {
                        "name": "JSESSIONID",
                        "value": csrf,
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": False,
                        "secure": True,
                    }
                ]
            )

        page = context.new_page()
        page.goto("https://www.linkedin.com/feed/", timeout=60000)

        for job in jobs:
            try:
                job_url = job.get("job_url", "https://www.linkedin.com/jobs/")
                page.goto(job_url, timeout=60000)
                time.sleep(1)
                easy_apply = page.locator("button:has-text('Easy Apply')").first
                if easy_apply.count() > 0:
                    if not dry_run:
                        easy_apply.click()
                        submit = page.locator("button:has-text('Submit application')").first
                        if submit.count() > 0:
                            submit.click()
                    report["applied"].append(job_url)

                recruiter_url = job.get("recruiter_profile")
                if recruiter_url:
                    page.goto(recruiter_url, timeout=60000)
                    time.sleep(1)
                    msg_btn = page.locator("button:has-text('Message')").first
                    if msg_btn.count() > 0:
                        if not dry_run:
                            msg_btn.click()
                            box = page.locator("div[role='textbox']").last
                            box.fill(
                                recruiter_texts.get(
                                    job.get("job_url", ""), "Hi, I applied and would value your consideration."
                                )
                            )
                            send = page.locator("button:has-text('Send')").first
                            if send.count() > 0:
                                send.click()
                        report["messaged"].append(recruiter_url)
            except Exception as exc:
                report["errors"].append(f"{job.get('job_url')}: {exc}")

        try:
            page.goto("https://www.linkedin.com/search/results/people/?network=%5B%22F%22%5D", timeout=60000)
            time.sleep(2)
            message_buttons = page.locator("button:has-text('Message')")
            if message_buttons.count() > 0:
                if not dry_run:
                    message_buttons.first.click()
                    box = page.locator("div[role='textbox']").last
                    box.fill(referral_note)
                    send = page.locator("button:has-text('Send')").first
                    if send.count() > 0:
                        send.click()
                report["referrals"].append("network_contact")
        except Exception as exc:
            report["errors"].append(f"network referral: {exc}")

        try:
            page.goto("https://www.linkedin.com/search/results/people/", timeout=60000)
            time.sleep(2)
            connect = page.locator("button:has-text('Connect')")
            if connect.count() > 0:
                if not dry_run:
                    connect.first.click()
                    add_note = page.locator("button:has-text('Add a note')").first
                    if add_note.count() > 0:
                        add_note.click()
                        note_box = page.locator("textarea#custom-message").first
                        note_box.fill(referral_note[:280])
                        send_invite = page.locator("button:has-text('Send')").first
                        if send_invite.count() > 0:
                            send_invite.click()
                report["referrals"].append("new_connection_requested")
        except Exception as exc:
            report["errors"].append(f"outside network referral: {exc}")

        context.close()
        browser.close()

    report["ok"] = len(report["errors"]) == 0
    return report


def auth_section() -> UserProfile:
    st.subheader("1) Account Creation / Sign In")
    login_type = st.radio("Choose sign-in provider", ["LinkedIn", "Google", "Manual"])

    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    current = load_profile(st.session_state.user_id) or UserProfile(id=st.session_state.user_id)

    if login_type == "LinkedIn":
        url = linkedin_auth_url()
        if url:
            st.link_button("Sign in with LinkedIn", url=url)
        else:
            st.info("Set LINKEDIN_CLIENT_ID and LINKEDIN_REDIRECT_URI for OAuth button.")
        upload = st.file_uploader("Upload LinkedIn profile export JSON", type=["json"])
        if upload is not None:
            imported = import_linkedin_profile_from_file(upload)
            for key, value in imported.items():
                setattr(current, key, value)
            st.success("LinkedIn data imported.")

    elif login_type == "Google":
        url = google_auth_url()
        if url:
            st.link_button("Sign in with Google", url=url)
        else:
            st.info("Set GOOGLE_CLIENT_ID and GOOGLE_REDIRECT_URI for OAuth button.")

    st.subheader("2) Complete your profile")
    current.name = st.text_input("Name", value=current.name)
    current.email = st.text_input("Email", value=current.email)
    current.location = st.text_input("Current Location", value=current.location)
    current.skills = st.text_area("Skills (comma separated)", value=current.skills)
    current.preferred_location = st.text_input("Preferred Location", value=current.preferred_location)
    current.projects = st.text_area("Projects", value=current.projects)
    current.education = st.text_area("Education", value=current.education)
    current.certifications = st.text_area("Certifications", value=current.certifications)
    current.years_experience = st.text_input("Years of Experience", value=current.years_experience)
    current.target_role = st.text_input("Primary Target Role", value=current.target_role)
    current.linkedin_url = st.text_input("LinkedIn Profile URL", value=current.linkedin_url)

    if st.button("Save Profile"):
        file_path = save_profile(current)
        st.success(f"Saved profile to {file_path}")

    return current


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="💼", layout="wide")
    st.title(APP_NAME)
    st.caption(HEADLINE)

    with st.sidebar:
        st.header("Configuration")
        st.write(f"Model: `{OLLAMA_MODEL}`")
        st.write(f"Ollama endpoint: `{OLLAMA_URL}`")
        if st.button("Test Ollama connection"):
            try:
                ping = requests.get(f"{OLLAMA_URL}/api/tags", timeout=15)
                ping.raise_for_status()
                st.success("Ollama reachable.")
            except requests.RequestException as exc:
                st.error(f"Ollama not reachable: {exc}")

    profile = auth_section()

    st.divider()
    st.header("Job Applying Agent")

    col1, col2, col3 = st.columns(3)
    role = col1.text_input("Job role", value=profile.target_role)
    geography = col2.text_input("Geography / City", value=profile.preferred_location or profile.location)
    count = col3.number_input("Number of jobs", min_value=1, max_value=25, value=10)
    remote_ok = st.checkbox("Include remote")

    if st.button("Scan jobs + build resume"):
        if not role.strip() or not geography.strip():
            st.error("Please enter role and geography first.")
            return
        with st.spinner("Discovering jobs with AI..."):
            jobs = discover_jobs(role=role, geography=geography, remote_ok=remote_ok, count=count)
        if not jobs:
            st.error("No jobs found from model response. Check Ollama connectivity/model output.")
            return
        st.session_state.jobs = jobs

        with st.spinner("Generating ATS resume..."):
            resume_md = build_resume(profile, jobs)
        st.session_state.resume_md = resume_md

        st.success(f"Prepared {len(jobs)} jobs and a tailored resume.")

    if "jobs" in st.session_state:
        st.subheader("Discovered Jobs")
        st.dataframe(st.session_state.jobs)

    if "resume_md" in st.session_state:
        st.subheader("Generated Resume (Markdown)")
        st.markdown(st.session_state.resume_md)
        st.download_button(
            "Download Resume .md",
            data=st.session_state.resume_md,
            file_name=f"resume_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    st.divider()
    st.header("Post-Apply Messaging + Referrals")

    if st.button("Generate recruiter messages") and "jobs" in st.session_state:
        recruiter_texts = {
            j.get("job_url", f"job-{idx}"): recruiter_message(profile, j)
            for idx, j in enumerate(st.session_state.jobs)
        }
        st.session_state.recruiter_texts = recruiter_texts
        st.success("Generated recruiter messages.")

    if "recruiter_texts" in st.session_state:
        for key, value in st.session_state.recruiter_texts.items():
            with st.expander(f"Message for {key}"):
                st.write(value)

    ask_ref = st.button("Get me referrals")
    if ask_ref:
        note = referral_message(profile, role)
        st.session_state.referral_note = note
        st.info(note)

    st.subheader("Run LinkedIn Browser Agent")
    dry_run = st.checkbox("Dry run (recommended first)", value=True)
    st.warning(
        "Automation depends on your LinkedIn session cookie and may break with UI changes. Ensure legal/compliance checks."
    )
    if st.button("Run agent: apply + message + referral"):
        if "jobs" not in st.session_state:
            st.error("Generate jobs first.")
            return
        recruiter_texts = st.session_state.get("recruiter_texts", {})
        referral_note = st.session_state.get("referral_note", "Hi! I hope you're well. Could you please refer me if relevant?")
        with st.spinner("Running browser automation..."):
            report = linkedin_agent_run(st.session_state.jobs, recruiter_texts, referral_note, dry_run=dry_run)
        st.json(report)


if __name__ == "__main__":
    main()
