# Security Policy

## Supported version

Only the latest commit on the default branch (`v1`) is supported.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead: open the **Security** tab of this repository and choose **Report a vulnerability**.

Include what you found, how to reproduce it, and what you think the impact is. You can expect a first reply within a week. This is a student research project, so there is no bug bounty.

## Scope notes

- The repository contains only synthetic data. No real customer data should ever be added.
- API keys belong in a local `.env` file, which is gitignored. If you find a key in the history, please report it so it can be revoked.
