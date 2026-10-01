# Team WTM Git and GitHub Workflow

This document defines the standard workflow for PM-1, PM-2, PM-3, and PM-4 when contributing to the Team WTM project.

The main goals are to:

- keep the `main` branch stable;
- make work easy to review;
- avoid conflicting changes;
- connect every meaningful task to a GitHub Issue;
- use Pull Requests before merging work;
- keep the Project board up to date.

---

## 1. Start With an Assigned GitHub Issue

Go to:

`marblehub/data-circle -> Issues`

Open the Issue assigned to you.

Read:

- the description;
- the task list;
- the acceptance criteria;
- the milestone;
- any comments or instructions.

When you actually start working, move the Issue on the Project board:

```text
Ready -> In Progress
```

Do not start major project work without a GitHub Issue.

---

## 2. Update Your Local `main` Branch

Before creating a new branch, always update your local `main`.

```bash
cd ~/redi-school_repo/data-circle

git switch main
git pull origin main
```

This ensures that your new branch starts from the latest version of the team project.

### Important

Always create a new branch from an updated `main`.

Do not create a new task branch from an old feature branch.

---

## 3. Create a Branch for the Issue

Use a branch name that includes the Issue number.

Recommended format:

```text
feature/<issue-number>-<short-description>
```

Example:

```bash
git switch -c feature/4-missing-values
```

Other examples:

```text
feature/5-target-distribution
feature/7-geospatial-analysis
feature/10-baseline-model
```

For documentation work:

```text
docs/8-update-data-dictionary
```

For bug fixes:

```text
fix/12-construction-year-handling
```

Avoid unclear branch names such as:

```text
pm2-branch
my-work
test
final
final-version
```

Branch names should describe the work, not the person.

---

## 4. Enter the Team WTM Project Folder

Most project work should be done inside:

```text
projects/water_pumps/team-WTM/
```

From the repository root:

```bash
cd projects/water_pumps/team-WTM
```

Install or update the project environment:

```bash
uv sync
```

Open VS Code:

```bash
code .
```

Team members should normally avoid changing files outside `team-WTM/` unless PM-1 has specifically coordinated that change.

---

## 5. Do the Assigned Work

Examples of responsibilities include:

### PM-1

- project coordination;
- GitHub/project management;
- integration;
- documentation;
- upstream synchronisation;
- shared technical work.

### PM-2

- data quality;
- missing-value analysis;
- preprocessing;
- data documentation.

### PM-3

- exploratory data analysis;
- visualisations;
- geospatial analysis;
- hypothesis testing.

### PM-4

- preprocessing pipeline;
- modelling;
- evaluation;
- model interpretation;
- dashboard support.

These roles are flexible. Team members may contribute outside their primary area.

### Jupyter Notebook Rule

Avoid editing the same notebook at the same time.

Jupyter notebooks can create difficult Git merge conflicts.

Where possible:

- use separate notebooks for separate tasks;
- move reusable code into `src/team_wtm/`.

---

## 6. Check Your Changes Before Committing

Before committing:

```bash
git status
git diff
```

Make sure you have not accidentally added:

- raw CSV files;
- ZIP files;
- generated datasets;
- `.venv/`;
- `.env`;
- secrets or passwords;
- unrelated ReDI files.

Raw project data should remain local and should not be committed to GitHub.

---

## 7. Commit Your Work

Stage your changes:

```bash
git add .
```

Check again:

```bash
git status
```

Then commit:

```bash
git commit -m "feat: analyse missing values"
```

Recommended commit prefixes:

```text
feat:      new functionality or analysis
fix:       bug or correction
docs:      documentation
test:      tests
refactor:  code restructuring
chore:     maintenance or setup
```

Examples:

```text
feat: add target distribution analysis
feat: add geospatial pump analysis
feat: create baseline classifier
fix: handle invalid construction years
docs: update data dictionary
refactor: move cleaning functions into data module
test: add preprocessing tests
```

Avoid vague commit messages such as:

```text
update
changes
stuff
final
final2
```

---

## 8. Push Your Branch

For the first push:

```bash
git push -u origin feature/4-missing-values
```

After the first push, later updates only need:

```bash
git push
```

Replace the example branch name with your actual branch name.

---

## 9. Create a Pull Request

Go to:

`marblehub/data-circle -> Pull requests -> New pull request`

Make sure the Pull Request is created inside the Team WTM fork.

Use:

```text
base repository: marblehub/data-circle
base branch:      main
compare branch:   your feature branch
```

Example:

```text
feature/4-missing-values
          ↓
        main
```

### Important

Do not normally create the Pull Request against:

```text
ReDI-School/data-circle
```

The ReDI repository is the upstream repository.

Team WTM work should normally be merged into:

```text
marblehub/data-circle:main
```

unless ReDI instructors specifically request an upstream Pull Request.

---

## 10. Write a Clear Pull Request Description

Use this structure:

```markdown
## Summary

Briefly explain what this Pull Request does.

## Changes

- list the main changes
- keep the list short and clear
- mention important files or analyses

## Validation

- explain how the work was checked
- confirm the notebook/script runs
- confirm no raw data was committed

## Notes for Reviewer

Mention anything the reviewer should pay special attention to.

Closes #<issue-number>
```

Example:

```markdown
## Summary

Analyses missing values in the water-pump dataset.

## Changes

- identified columns with missing values
- calculated missing-value percentages
- documented important observations

## Validation

- notebook runs successfully
- results were checked against the source data
- no raw data was committed

## Notes for Reviewer

Please check whether the missing-value handling recommendations are clear.

Closes #4
```

The line:

```text
Closes #4
```

links the Pull Request to Issue #4.

When the Pull Request is merged into `main`, GitHub should automatically close the linked Issue.

---

## 11. Move the Work to Review

When the Pull Request is opened, move the Issue on the Project board:

```text
In Progress -> Review
```

The Pull Request card can also be moved to:

```text
Review
```

---

## 12. Request a Peer Review

Do not merge immediately.

Another team member should review the Pull Request.

Reviewers should check:

- correctness;
- clarity;
- reproducibility;
- documentation;
- project structure;
- unnecessary duplication;
- whether the Issue acceptance criteria were met.

Suggested review rotation:

```text
PM-1 -> PM-2
PM-2 -> PM-3
PM-3 -> PM-4
PM-4 -> PM-1 or PM-2
```

This is flexible depending on availability and technical relevance.

---

## 13. Respond to Review Comments

If changes are requested, stay on the same branch.

Do not create another branch.

Make the requested changes and then:

```bash
git add .
git commit -m "fix: address review comments"
git push
```

The existing Pull Request will update automatically.

Continue until the reviewer approves the Pull Request.

---

## 14. Merge the Pull Request

After approval, use:

```text
Squash and merge
```

This keeps the `main` branch history clean.

For example, several small commits such as:

```text
feat: add missing-value analysis
fix: correct percentage calculation
docs: clarify findings
```

can become one clean commit on `main`:

```text
feat: analyse missing values (#4)
```

Delete the remote feature branch after merging if GitHub offers the option.

---

## 15. Update Your Local Repository After Merging

Return to the repository root:

```bash
cd ~/redi-school_repo/data-circle
```

Switch back to `main`:

```bash
git switch main
```

Update it:

```bash
git pull origin main
```

Delete the old local branch:

```bash
git branch -d feature/4-missing-values
```

Clean up old remote branch references:

```bash
git fetch --prune
```

Replace the example branch name with your actual branch.

---

## 16. Move the Project Card to Done

After the Pull Request is merged:

```text
Review -> Done
```

If the PR contains:

```text
Closes #4
```

GitHub should also automatically close Issue #4.

---

# Complete Team Workflow

```text
Assigned GitHub Issue
        ↓
Ready
        ↓
Start work
        ↓
In Progress
        ↓
git switch main
        ↓
git pull origin main
        ↓
Create new Issue branch
        ↓
Do the work
        ↓
git status
        ↓
git diff
        ↓
git add .
        ↓
git commit
        ↓
git push
        ↓
Open Pull Request
        ↓
Target marblehub/data-circle:main
        ↓
Move to Review
        ↓
Peer review
        ↓
Address comments if needed
        ↓
Approval
        ↓
Squash and merge
        ↓
Issue automatically closes
        ↓
Pull latest main
        ↓
Delete old branch
        ↓
Done
```

---

# Golden Rule

Before starting every new Issue:

```bash
git switch main
git pull origin main
```

Then create a new branch:

```bash
git switch -c feature/<issue-number>-<description>
```

Never start a new task from an old feature branch.

---

# PM-1 Additional Responsibilities

PM-1 follows the same branch, Issue, Pull Request, and review workflow as all other team members.

In addition, PM-1 is responsible for:

- maintaining the GitHub Project board;
- organising the Sprint backlog;
- assigning Issues;
- checking blockers;
- monitoring Pull Requests;
- coordinating sprint deliverables;
- keeping documentation consistent;
- synchronising the team fork with the ReDI upstream repository.

PM-1 should not routinely bypass the Pull Request process.

---

# Synchronising With the ReDI Upstream Repository

The Team WTM fork uses:

```text
origin -> marblehub/data-circle
```

The original ReDI repository uses:

```text
upstream -> ReDI-School/data-circle
```

PM-1 normally coordinates upstream synchronisation.

From the repository root:

```bash
cd ~/redi-school_repo/data-circle

git switch main
git fetch upstream
```

Check whether ReDI has new commits:

```bash
git log main..upstream/main --oneline
```

If updates exist:

```bash
git merge upstream/main
git push origin main
```

Other team members can then update normally using:

```bash
git switch main
git pull origin main
```

---

# Project Board Status

Team WTM uses the following status flow:

```text
Backlog
   ↓
Ready
   ↓
In Progress
   ↓
Review
   ↓
Done
```

### Backlog

Tasks identified for future work.

### Ready

Tasks prepared and ready for someone to start.

### In Progress

A team member is actively working on the Issue.

### Review

A Pull Request has been opened and is waiting for review or approval.

### Done

The Pull Request has been merged and the Issue is complete.

---

# Important Team Rules

1. Do not normally work directly on `main`.

2. Every meaningful task should have a GitHub Issue.

3. Every Issue should normally have one primary owner.

4. Create a new branch for each Issue.

5. Always create the branch from an updated `main`.

6. Do not commit raw datasets, passwords, secrets, or `.env` files.

7. Avoid editing the same Jupyter notebook simultaneously.

8. Use Pull Requests to merge work into `main`.

9. Every meaningful Pull Request should receive peer review.

10. Use `Squash and merge` after approval.

11. Keep the Project board status accurate.

12. Keep most Team WTM changes inside:

```text
projects/water_pumps/team-WTM/
```

13. Do not create Pull Requests into `ReDI-School/data-circle` unless specifically requested by the ReDI instructors.

14. Ask the team before making major structural changes that may affect everyone.

15. Keep code, analysis, and documentation clear enough for another team member to understand and reproduce.