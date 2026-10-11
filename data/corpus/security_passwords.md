# Password and account security

Northstar accounts use SSO. You should almost never set a separate password for an internal tool.

## Password manager

1Password Business is mandatory. Do not store work passwords in the browser, Notes, or Slack.

## MFA

Phishing-resistant MFA (security key or company SSO) is required. SMS MFA is not accepted for production cloud consoles.

## Sharing

Never share passwords or API keys in chat. If a key leaked, revoke it in the vendor console and file an incident in `#sec-incidents` within 1 hour.

## Laptop lock

Lock the screen when you leave a desk. Unattended unlocked laptops in the office are a security write-up after the second occurrence.

## Personal devices

Personal laptops may not clone Northstar repositories. Personal phones may use Slack and email with MDM. The internal assistant API is not supported on unmanaged devices.
