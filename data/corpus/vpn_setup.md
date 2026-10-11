# VPN setup

All access to Northstar private services from outside the office network must use the company VPN.

## Who needs it

You need the VPN for Git, the staging cluster, the internal FastAPI assistant, and shared drives. Public marketing sites do not require VPN.

## Client

Install the official WireGuard profile from the IT portal. Do not use consumer VPN apps on a work laptop.

## First-time setup

1. Enroll the laptop in MDM.
2. Download the profile named `northstar-staff`.
3. Enable the tunnel before opening internal URLs.
4. Sign in with your company Google account (SSO). Password login to VPN is disabled.

## Split tunnel

The profile routes only RFC1918 and `*.northstar.internal` through the tunnel. General web traffic stays on your ISP.

## Troubleshooting

If Git times out, confirm the tunnel shows Connected and that the system clock is correct. After 3 failed SSO attempts the account locks for 15 minutes. IT hours are 08:00–18:00 GST on working days via `#it-help`.

Mobile devices may use the same profile. Personal phones are allowed for Slack and email only; do not clone repositories onto a personal phone.
