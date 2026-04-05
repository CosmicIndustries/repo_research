"""
github_write.py — GitHub write operations: issues, PRs, file edits, branch creation.
All operations require a PAT with appropriate scopes.
"""

import base64
import time
from typing import Optional

import requests


class GitHubWriter:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        if not token:
            raise ValueError("A GitHub PAT is required for write operations.")
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "github-research-assistant",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _req(self, method: str, path: str, **kwargs) -> dict:
        url  = f"{self.BASE}{path}"
        resp = self.session.request(method, url, timeout=20, **kwargs)
        if resp.status_code == 403:
            raise PermissionError(f"403 — check PAT scopes. {resp.json().get('message','')}")
        if resp.status_code == 404:
            raise FileNotFoundError(f"404 — {path}")
        if resp.status_code == 422:
            raise ValueError(f"422 Unprocessable — {resp.json()}")
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── Issues ─────────────────────────────────────────────────────────────────

    def create_issue(self, owner: str, repo: str, title: str, body: str,
                     labels: list[str] = [], assignees: list[str] = []) -> dict:
        payload = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        result = self._req("POST", f"/repos/{owner}/{repo}/issues", json=payload)
        return {
            "number":   result["number"],
            "title":    result["title"],
            "html_url": result["html_url"],
            "state":    result["state"],
        }

    def list_issues(self, owner: str, repo: str, state: str = "open") -> list[dict]:
        result = self._req("GET", f"/repos/{owner}/{repo}/issues",
                           params={"state": state, "per_page": 50})
        return [{"number": i["number"], "title": i["title"],
                 "state": i["state"], "html_url": i["html_url"]} for i in result]

    def comment_issue(self, owner: str, repo: str, number: int, body: str) -> dict:
        result = self._req("POST", f"/repos/{owner}/{repo}/issues/{number}/comments",
                           json={"body": body})
        return {"id": result["id"], "html_url": result["html_url"]}

    # ── Branches ───────────────────────────────────────────────────────────────

    def get_default_branch_sha(self, owner: str, repo: str) -> tuple[str, str]:
        """Returns (branch_name, sha)."""
        info   = self._req("GET", f"/repos/{owner}/{repo}")
        branch = info["default_branch"]
        ref    = self._req("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        return branch, ref["object"]["sha"]

    def create_branch(self, owner: str, repo: str, branch: str, from_sha: str) -> dict:
        return self._req("POST", f"/repos/{owner}/{repo}/git/refs",
                         json={"ref": f"refs/heads/{branch}", "sha": from_sha})

    def branch_exists(self, owner: str, repo: str, branch: str) -> bool:
        try:
            self._req("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
            return True
        except FileNotFoundError:
            return False

    # ── File edits ─────────────────────────────────────────────────────────────

    def get_file(self, owner: str, repo: str, path: str,
                 branch: Optional[str] = None) -> dict:
        params = {"ref": branch} if branch else {}
        result = self._req("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)
        content = base64.b64decode(result["content"]).decode("utf-8", errors="replace")
        return {"sha": result["sha"], "content": content, "path": result["path"],
                "html_url": result["html_url"]}

    def update_file(self, owner: str, repo: str, path: str, content: str,
                    message: str, file_sha: str, branch: str) -> dict:
        encoded = base64.b64encode(content.encode()).decode()
        result  = self._req("PUT", f"/repos/{owner}/{repo}/contents/{path}", json={
            "message": content,
            "content": encoded,
            "sha":     file_sha,
            "branch":  branch,
            "message": message,
        })
        return {
            "commit_sha": result["commit"]["sha"],
            "html_url":   result["content"]["html_url"],
        }

    def create_file(self, owner: str, repo: str, path: str, content: str,
                    message: str, branch: str) -> dict:
        encoded = base64.b64encode(content.encode()).decode()
        result  = self._req("PUT", f"/repos/{owner}/{repo}/contents/{path}", json={
            "message": message,
            "content": encoded,
            "branch":  branch,
        })
        return {
            "commit_sha": result["commit"]["sha"],
            "html_url":   result["content"]["html_url"],
        }

    # ── Pull Requests ──────────────────────────────────────────────────────────

    def create_pr(self, owner: str, repo: str, title: str, body: str,
                  head: str, base: str, draft: bool = False) -> dict:
        result = self._req("POST", f"/repos/{owner}/{repo}/pulls", json={
            "title": title,
            "body":  body,
            "head":  head,
            "base":  base,
            "draft": draft,
        })
        return {
            "number":   result["number"],
            "title":    result["title"],
            "html_url": result["html_url"],
            "state":    result["state"],
            "draft":    result.get("draft", False),
        }

    def list_prs(self, owner: str, repo: str, state: str = "open") -> list[dict]:
        result = self._req("GET", f"/repos/{owner}/{repo}/pulls",
                           params={"state": state, "per_page": 30})
        return [{"number": r["number"], "title": r["title"],
                 "state": r["state"], "html_url": r["html_url"],
                 "head": r["head"]["ref"], "base": r["base"]["ref"]} for r in result]

    # ── High-level helpers ─────────────────────────────────────────────────────

    def patch_file_and_pr(self, owner: str, repo: str, file_path: str,
                          new_content: str, branch_name: str,
                          commit_msg: str, pr_title: str, pr_body: str,
                          draft: bool = True) -> dict:
        """
        Full workflow: create branch → update file → open PR.
        Returns combined result dict.
        """
        default_branch, sha = self.get_default_branch_sha(owner, repo)

        # Create branch (or reuse if already exists)
        if not self.branch_exists(owner, repo, branch_name):
            self.create_branch(owner, repo, branch_name, sha)
            time.sleep(0.5)

        # Get current file SHA for update
        current = self.get_file(owner, repo, file_path, branch=default_branch)
        update  = self.update_file(owner, repo, file_path, new_content,
                                   commit_msg, current["sha"], branch_name)

        # Open PR
        pr = self.create_pr(owner, repo, pr_title, pr_body,
                            head=branch_name, base=default_branch, draft=draft)
        return {
            "branch":     branch_name,
            "commit_sha": update["commit_sha"],
            "pr":         pr,
        }

    def create_vuln_issue(self, owner: str, repo: str, finding: dict) -> dict:
        """Open a security issue from a scanner Finding dict."""
        sev   = finding["severity"]
        title = f"[{sev}] {finding['title']} in `{finding['path']}`"
        body  = (
            f"## {finding['title']}\n\n"
            f"**Severity:** `{sev}`  \n"
            f"**CWE:** [{finding['cwe']}](https://cwe.mitre.org/data/definitions/"
            f"{finding['cwe'].replace('CWE-','')}.html)  \n"
            f"**File:** [`{finding['path']}`]({finding['url']})  \n\n"
            f"### Description\n{finding['detail']}\n\n"
            f"### Matched Pattern\n```\n{finding['match'][:500]}\n```\n\n"
            f"### Context Line\n```\n{finding['line_hint']}\n```\n\n"
            f"---\n*Reported by GitHub Research Assistant — white-hat scan.*"
        )
        return self.create_issue(owner, repo, title, body,
                                 labels=["security", sev.lower()])
