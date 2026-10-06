"""Seed the database with sample admission posts.

    uv run python scripts/seed.py

⚠️  THE CONTENT BELOW IS SYNTHETIC PLACEHOLDER DATA.  ⚠️

Every figure, deadline, GPA threshold and waiver percentage here is INVENTED.
It exists so the pipeline and the smoke test can be built and run before real
content is available. It is NOT DIU admission information and must never be
presented as such.

Before making any claim about retrieval accuracy, replace these with real
content copied from the official DIU admission site, and replace the question
set in tests/test_retrieval_smoke.py with questions real applicants ask.
Synthetic data will not tell you whether retrieval works on real admission
prose, which is full of program codes, waiver tables and abbreviations that
behave nothing like generated text.
"""

import sys

from app.database.connection import get_conn

SEED_MARKER = "[SAMPLE DATA — NOT REAL DIU INFORMATION]"

CATEGORIES = ["Admission Requirements", "Fees", "Deadlines", "Scholarships", "Programs"]

POSTS: list[tuple[str, str]] = [
    (
        "CSE Undergraduate Admission Requirements, Fall 2026",
        """Applicants to the B.Sc. in Computer Science and Engineering must have
completed HSC or an equivalent examination with a minimum combined GPA of 8.00
across SSC and HSC, with at least GPA 3.50 in each.

Mathematics at the HSC level is mandatory for all engineering programs.
Applicants from a science background who did not take Mathematics are not
eligible for the CSE program and should consider the BBA or English programs.

Applicants holding GCE A-Level qualifications must have passed at least two
subjects including Mathematics, with no grade below C. O-Level applicants must
have passed at least five subjects.

Diploma holders in Computer Technology from a recognised polytechnic may apply
for lateral entry into the third semester, subject to credit evaluation by the
department.""",
    ),
    (
        "Tuition Fees and Payment Schedule for CSE, Fall 2026",
        """The per-credit tuition fee for the B.Sc. in Computer Science and
Engineering is BDT 6,500 for the Fall 2026 intake. The programme comprises 148
credits in total across 12 semesters.

A one-time admission fee of BDT 25,000 is payable at the time of enrolment.
This fee is non-refundable once registration is complete.

Semester fees are payable in three instalments. The first instalment is due at
registration, the second before the mid-term examination, and the third before
the final examination. A late payment surcharge of BDT 1,000 applies after each
deadline.

Laboratory and library fees total BDT 8,000 per semester for engineering
programmes. Students who withdraw before the add-drop deadline receive a full
refund of semester tuition, less the admission fee.""",
    ),
    (
        "Application Deadlines for Fall 2026 Intake",
        """Online applications for the Fall 2026 semester open on 1 June 2026 and
close on 15 August 2026 at 11:59 PM.

The admission test for engineering programmes will be held on 22 August 2026.
Admit cards will be available for download from 18 August 2026.

Results will be published on 28 August 2026. Selected applicants must complete
enrolment and pay the admission fee by 8 September 2026, after which the seat
is released to the waiting list.

Classes for the Fall 2026 semester begin on 15 September 2026. Late enrolment
is possible until 20 September 2026 with a late fee of BDT 2,000.""",
    ),
    (
        "Merit and Need-Based Scholarships and Waivers",
        """Applicants with GPA 5.00 in both SSC and HSC receive a 100% tuition
waiver for the first semester. The waiver continues for subsequent semesters if
the student maintains a CGPA of 3.75 or above.

Applicants with a combined GPA between 9.00 and 9.99 receive a 50% waiver for
the first semester, and those between 8.00 and 8.99 receive a 25% waiver.

Siblings enrolled simultaneously each receive a 15% tuition waiver for the
duration of their overlapping enrolment.

Children of freedom fighters are eligible for a 100% tuition waiver throughout
the programme, subject to submission of a valid certificate.

Need-based financial aid is assessed separately by the Student Welfare office.
Applications require documentary evidence of family income and are considered
once per academic year.""",
    ),
    (
        "Required Documents for Admission",
        """All applicants must submit attested photocopies of the SSC and HSC
certificates and mark sheets at the time of enrolment. Original certificates
must be presented for verification and will be returned immediately.

Four recent passport-sized photographs with a white background are required.

A photocopy of the applicant's National ID card, or birth certificate if the
applicant does not yet hold a National ID, must be provided.

Applicants transferring from another university must submit a transcript,
a certificate of good standing, and a course syllabus for credit evaluation.

Foreign applicants must additionally provide an equivalence certificate issued
by the University Grants Commission of Bangladesh.""",
    ),
    (
        "BBA Program Admission Requirements",
        """Applicants to the Bachelor of Business Administration must have a
minimum combined GPA of 7.00 across SSC and HSC, with no individual GPA below
3.00.

Unlike the engineering programmes, HSC Mathematics is not mandatory for BBA.
Applicants from humanities, commerce and science backgrounds are all eligible.

The BBA programme comprises 126 credits across 8 semesters. The per-credit fee
is BDT 5,500 for the Fall 2026 intake.

The admission test for business programmes covers English, general mathematics
and analytical ability, and is held on 23 August 2026.""",
    ),
    (
        "Admission Test Format and Syllabus",
        """The admission test for engineering programmes is a 90-minute written
examination consisting of 100 multiple-choice questions.

The distribution is Mathematics 40 marks, Physics 30 marks, English 20 marks,
and general knowledge 10 marks. There is a negative marking of 0.25 for each
incorrect answer.

The pass mark is 40. Candidates scoring below 40 are not eligible for
admission regardless of their SSC and HSC results.

The syllabus follows the national HSC curriculum. No questions are set from
outside the HSC syllabus.

Calculators are not permitted in the examination hall. Mobile phones and smart
watches must be deposited before entry.""",
    ),
    (
        "Hostel and Accommodation Facilities",
        """Separate hostel accommodation is available for male and female
students. Seats are allocated on a first-come, first-served basis after
enrolment is complete.

The hostel fee is BDT 4,500 per month for a shared room accommodating three
students, and BDT 7,500 per month for a twin-sharing room.

A refundable security deposit of BDT 5,000 is payable at the time of hostel
allocation.

Meal charges are billed separately at approximately BDT 4,000 per month.
Students may opt out of the meal plan.

Hostel residents must observe the 10:00 PM curfew. Overnight absence requires
written permission from the hostel superintendent.""",
    ),
    (
        "Credit Transfer and Lateral Entry Policy",
        """Students transferring from another recognised university may apply for
credit transfer of up to 50% of the total programme credits.

Only courses in which the applicant obtained a grade of B or above are eligible
for transfer. Courses older than five years are not considered.

The departmental credit evaluation committee reviews each application. The
decision of the committee is final.

Diploma holders entering laterally are typically granted 30 to 40 credits
depending on the diploma discipline and the department's assessment.

Credit transfer applications must be submitted within the first two weeks of
the semester of enrolment.""",
    ),
    (
        "International Student Admission Procedure",
        """International applicants may apply online without appearing for the
written admission test. Selection is based on academic records and an online
interview.

Applicants must hold qualifications equivalent to the Bangladeshi HSC, verified
by an equivalence certificate from the University Grants Commission.

A student visa is required before enrolment. The university issues an
acceptance letter to support the visa application once the applicant has paid
the admission fee.

International students pay tuition at the rate of USD 120 per credit for
engineering programmes and USD 100 per credit for business programmes.

Health insurance covering the duration of study is mandatory for all
international students.""",
    ),
    (
        "Refund Policy for Withdrawn Admissions",
        """Students who withdraw before the commencement of classes receive a
refund of 100% of the semester tuition. The admission fee is not refundable
under any circumstance.

Withdrawal within the first two weeks of classes attracts a refund of 75% of
semester tuition.

Withdrawal between the third and fourth week attracts a refund of 50%.

No refund of semester tuition is available after the fourth week of classes.

Refund requests must be submitted in writing to the Registrar's office.
Processing takes approximately four to six weeks.""",
    ),
    (
        "Evening and Weekend Program Options",
        """The university offers evening programmes in Business Administration
and Computer Science for working professionals.

Evening classes are held from 6:00 PM to 9:30 PM on weekdays. Weekend
programmes run on Friday and Saturday.

The per-credit fee for evening programmes is BDT 7,000, higher than the day
programme rate, reflecting the smaller cohort size.

Applicants to evening programmes must have at least two years of documented
work experience.

Evening students are not eligible for merit-based tuition waivers, but may
apply for the instalment payment plan.""",
    ),
    (
        "Departmental Contact Information for Admission Queries",
        """The central admission office is located on the ground floor of the
Administrative Building and is open from 9:00 AM to 5:00 PM, Sunday through
Thursday.

Admission-related queries may be sent to the admission helpdesk. Response time
is typically one working day.

The CSE department admission coordinator is available for programme-specific
queries on Monday and Wednesday afternoons.

During the peak admission period from June to September, the admission office
also operates on Saturdays from 10:00 AM to 2:00 PM.

Applicants requiring assistance with the online application form may visit the
IT helpdesk on the second floor.""",
    ),
    (
        "Attendance and Academic Progression Rules",
        """A minimum of 70% attendance is required in each course to be eligible
to sit the final examination.

Students falling below 70% attendance receive a grade of F for that course
regardless of their coursework performance.

A CGPA of at least 2.00 must be maintained for academic progression. Students
falling below this threshold are placed on academic probation for one semester.

Two consecutive semesters on probation result in dismissal from the programme.

Medical exemptions from the attendance requirement require documentation from
a registered medical practitioner submitted within seven days.""",
    ),
    (
        "Online Application Process, Step by Step",
        """Applicants begin by creating an account on the admission portal using
a valid email address and mobile number. A verification code is sent to both.

The application form requires personal details, academic records, and the
choice of up to three programmes in order of preference.

Scanned copies of the SSC and HSC mark sheets must be uploaded in PDF or JPEG
format, each under 2 MB.

The application fee of BDT 1,000 is payable online via mobile financial
services or card. The application is not submitted until payment is confirmed.

After submission, applicants receive an application number by SMS. This number
is required to download the admit card and to check results.""",
    ),
]


def main() -> int:
    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        if existing:
            print(f"Database already has {existing} post(s); not re-seeding.")
            print("To start clean:  dropdb -h localhost -p 5433 diu_admission "
                  "&& bash scripts/setup_db.sh")
            return 0

        user_id = conn.execute(
            """
            INSERT INTO users (name, email, password, status)
            VALUES ('Admission Officer (seed)', 'seed@example.invalid',
                    'NOT-A-REAL-HASH-seed-only', 'active')
            RETURNING id
            """
        ).fetchone()[0]

        category_ids = [
            conn.execute(
                "INSERT INTO categories (title, created_by) VALUES (%s, %s) RETURNING id",
                (title, user_id),
            ).fetchone()[0]
            for title in CATEGORIES
        ]

        for i, (title, body) in enumerate(POSTS):
            marked_body = f"{SEED_MARKER}\n\n{body.strip()}"
            post_id = conn.execute(
                """
                INSERT INTO posts (title, body, created_by, status, published_at,
                                   approval_status, approved_at, approved_by,
                                   verify_status, verified_at, verified_by)
                VALUES (%s, %s, %s, 'verified', NOW(),
                        'approved', NOW(), %s,
                        'verified', NOW(), %s)
                RETURNING id
                """,
                (title, marked_body, user_id, user_id, user_id),
            ).fetchone()[0]

            conn.execute(
                """
                INSERT INTO category_post (category_id, post_id, created_by)
                VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
                """,
                (category_ids[i % len(category_ids)], post_id, user_id),
            )

        print(f"Seeded 1 user, {len(CATEGORIES)} categories, {len(POSTS)} posts.")
        print("All posts are approved + verified and ready to ingest:")
        print("    uv run python -m app.ingestion.run --all-eligible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
