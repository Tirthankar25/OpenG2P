# domain_knowledge.py — 
# TO ADD A NEW DOMAIN REPO: add one entry to DOMAIN_REPOS below with a
# unique lowercase key, a short title, and a facts string. Nothing else in
# this file — or in llama_llm.py — needs to change.

DOMAIN_REPOS = {
    "smam": {
        "title": "SMAM (Sub-Mission on Agricultural Mechanization)",
        "facts": """
SMAM — the Sub-Mission on Agricultural Mechanization, a Government of
India Centrally Sponsored Scheme (under Rashtriya Krishi Vikas Yojana,
running since 2014-15, implemented through State/UT governments) that
subsidizes farm machinery for farmers — especially small, marginal, and
women farmers — via direct machine-purchase subsidies, Custom Hiring
Centres (CHCs), Farm Machinery Banks (FMBs), and hi-tech hubs.

Key facts (approximate — figures and subsidy rates vary by year, state,
and machine type; for anything precise and current, tell the user to
confirm at agrimachinery.nic.in or their state agriculture department
rather than stating an exact number with full confidence):
  - Goal: "reaching the unreached" — extend farm mechanization to small
    and marginal farmers, with particular emphasis on women farmers.
  - Main components: subsidized direct purchase of machinery by
    individual farmers; Custom Hiring Centres (CHCs) that rent out
    machinery; Farm Machinery Banks (FMBs) for shared equipment among
    groups; hi-tech hubs for high-value machinery; farmer training and
    demonstrations.
  - Subsidy is typically around 40-50% for general-category farmers, and
    higher (roughly 50-80%) for SC/ST, women, and small/marginal farmers,
    varying by machine type and state.
  - A group (FPO/SHG/cooperative) setting up a CHC or FMB can generally
    get subsidy support up to roughly ₹10 lakh, subject to state rules.
  - Cumulatively since inception (as reported by the Ministry of
    Agriculture): several thousand crore rupees released, over two
    million subsidized machines distributed, and tens of thousands of
    CHCs, FMBs, and hi-tech hubs established nationally. Farm power
    availability has risen substantially as mechanization has spread.
  - To apply, farmers generally use agrimachinery.nic.in or their state
    agriculture department's portal, submitting Aadhaar, land records,
    bank details, and a dealer quotation; caste/gender certificates
    unlock the higher subsidy tiers.
""".strip(),
    },

    "pm_kisan": {
        "title": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
        "facts": """
PM-KISAN — the Pradhan Mantri Kisan Samman Nidhi, a Government of India
Central Sector Scheme (100% funded by the Union government, unlike a
state-cost-shared Centrally Sponsored Scheme such as SMAM) launched in
December 2018 that gives landholding farmer families direct income
support by bank transfer, administered by the Department of Agriculture
and Farmers Welfare via pmkisan.gov.in.

Key facts (approximate — installment timing, beneficiary counts, and
exact exclusion rules can change; for anything precise and current, tell
the user to confirm at pmkisan.gov.in rather than stating an exact number
with full confidence):
  - Benefit: ₹6,000 per year, paid in three equal installments of ₹2,000
    each roughly every four months, by direct bank transfer (DBT) into
    the farmer's bank account.
  - Eligibility: all landholding farmer families qualify today — the
    scheme launched in Dec 2018 with a 2-hectare landholding cap, but
    that cap was removed in a Feb 2019 expansion, so land size is no
    longer a limit by itself. "Family" means husband, wife, and minor
    children.
  - Excluded categories: institutional landholders; current or former
    holders of constitutional posts; current or former MPs, MLAs, or
    municipal officials; government employees and pensioners with a
    monthly pension above roughly ₹10,000 (barring Group D/Class IV
    staff); anyone who paid income tax in the last assessment year; and
    practicing professionals (doctors, engineers, lawyers, chartered
    accountants, architects) registered with their professional body.
  - eKYC (electronic know-your-customer) is mandatory, and the
    beneficiary's bank account must be Aadhaar-seeded for the transfer to
    go through; payments are withheld for suspected exclusion-category
    cases pending verification.
  - Scale: roughly 9-10 crore farmers typically receive each installment
    (about 9.45 crore as of the mid-2026 installment, per the official
    site) — this shifts every cycle, so treat it as an order of
    magnitude, not a current count.
  - To apply: register via pmkisan.gov.in ("Farmers Corner" → "New
    Farmer Registration"), through a Common Service Centre (CSC), or via
    the local revenue/agriculture department's nodal officer, with
    Aadhaar, land ownership records, and Aadhaar-linked bank account
    details. Beneficiary status can be checked on the same site using an
    Aadhaar number, mobile number, or bank account number.
""".strip(),
    },

    # Add domains 3-6 here as you build them out, following the same shape:
    # "<repo_key>": {"title": "<Scheme Name (expansion)>", "facts": "..."},
}


def list_domain_keys() -> list:
    """The registered repo ids, in insertion order — used to build the
    search_scheme_docs tool's `repo` enum in llama_llm.py."""
    return list(DOMAIN_REPOS.keys())


def list_domain_titles() -> list:
    """Human-readable titles, in the same order — used to build the
    scope-restriction text in llama_llm.py so it stays in sync with
    whatever's registered here without any edits to that file."""
    return [entry["title"] for entry in DOMAIN_REPOS.values()]


def search_scheme_docs(repo: str, query: str = "") -> str:
    """Tool-facing entrypoint for the agent's search_scheme_docs tool call.

    Never raises — on an unrecognized repo it returns a clear "not
    registered" message (listing what IS available) so the model can
    fall back to the scope-restriction refusal instead of guessing.

    `query` is accepted (and logged) but not yet used for ranking/
    filtering, since each repo's "search" is currently its whole fact
    sheet — once real retrieval replaces that, this is the signature to
    keep; only the body changes.
    """
    key = (repo or "").strip().lower()
    print(f"[DOMAIN] search_scheme_docs(repo={key!r}, query={query!r})")

    entry = DOMAIN_REPOS.get(key)
    if not entry:
        known = ", ".join(list_domain_keys()) or "(none registered)"
        msg = f"No knowledge repo named '{repo}' is registered. Available repos: {known}."
        print(f"[DOMAIN] {msg}")
        return msg

    return f"{entry['title']} — reference facts:\n{entry['facts']}"
