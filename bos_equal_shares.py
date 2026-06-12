"""
An implementation of the algorithms found in:
"Method of Equal Shares with Bounded Overspending"
https://www.ac.tuwien.ac.at/comsoc2025/comsoc2025-papers/50.pdf

Programmer: Ivan Gorbachev
Date: 17/04/2026
"""
import logging
import math

from pabutools.analysis.cohesiveness import cohesive_groups
from scipy.optimize import root_scalar

from pabutools.election import Project, Instance, ApprovalBallot, ApprovalProfile, CardinalBallot, CardinalProfile


def get_utility(voter, project):
    if type(voter) == CardinalBallot:
        return voter.utility(project)
    return 1 if project in voter else 0


def bos_equal_shares(instance, profile):
    if not isinstance(profile, (ApprovalProfile, CardinalProfile)):
        raise TypeError("profile must be an instance of ApprovalProfile or CardinalProfile")
    if any(not isinstance(voter, (ApprovalBallot, CardinalBallot)) for voter in profile):
        raise TypeError("All items inside the profile must be ApprovalBallot or CardinalBallot instances")

    logger = logging.getLogger(__name__)
    logger.info("\nBOS equal shares")

    voters = list(profile)
    selected_projects = list()
    cost_selected_projects = 0

    budget = instance.budget_limit
    num_voters = profile.num_ballots()

    virtual_budgets = [budget / num_voters for _ in voters]
    all_projects = list(instance)

    logger.info(f"Budget: {budget}")
    logger.info(f"Virtual budgets: {[{str(v): round(b, 2)} for v, b in zip(voters, virtual_budgets)]}")

    budget_for_project = {
        project: sum(virtual_budgets[i] * get_utility(voter, project) for i, voter in enumerate(voters)) for
        project in all_projects}

    available_projects = [project for project in all_projects if cost_selected_projects + project.cost <= budget and
                          budget_for_project[project] > 0 and project not in selected_projects]

    while available_projects and cost_selected_projects < budget:
        logger.info(f"Remaining budget: {budget - cost_selected_projects}")
        best_alpha = 1
        best_rho = math.inf
        best_project = None

        project_metrics = {}

        for project in available_projects:
            supporters = [(i, voter) for i, voter in enumerate(voters) if get_utility(voter, project) > 0]
            if not supporters:
                continue
            supporters_budgets = [virtual_budgets[i] for i, voter in supporters]
            supporters_utils = [get_utility(voter, project) for i, voter in supporters]
            if sum(supporters_budgets) < project.cost:
                lambda_prime = math.inf
            else:
                res = root_scalar(
                    lambda lmbda: sum(min(b, lmbda * project.cost * u) for b, u in
                                      zip(supporters_budgets, supporters_utils)) - project.cost,
                    bracket=[0, 1.0]
                )
                lambda_prime = res.root
            lambdas = [virtual_budgets[i] / (project.cost * u) for i, u in
                       zip([s[0] for s in supporters], supporters_utils)]
            lambdas.append(lambda_prime)

            p_best_metric = math.inf
            p_best_rho = math.inf
            p_best_alpha = 1

            for lamb in lambdas:
                total_collected = (sum(
                    min(virtual_budgets[i], lamb * project.cost * u) for i, u in
                    zip([s[0] for s in supporters], supporters_utils)))
                alpha = min(total_collected / project.cost, 1)
                if alpha <= 0:
                    continue
                rho = lamb / alpha
                if rho / alpha < p_best_metric:
                    p_best_metric = rho / alpha
                    p_best_rho = rho
                    p_best_alpha = alpha

            if p_best_metric != math.inf:
                project_metrics[project] = (p_best_rho, p_best_alpha, p_best_metric)
                if p_best_metric < (best_rho / best_alpha if best_project else math.inf):
                    best_rho = p_best_rho
                    best_alpha = p_best_alpha
                    best_project = project

        if project_metrics:
            logger.info("Candidate projects evaluation details:")
            for proj, (r, a, m) in project_metrics.items():
                logger.info(f"  -> Project {proj.name}: rho = {r:.4f}, alpha = {a:.4f}, rho/alpha = {m:.4f}")

        if best_project is None:
            break

        logger.info(f"Selected project: {best_project}")
        logger.info(f"alpha = {best_alpha:.4f}, rho = {best_rho:.4f}, rho/alpha = {best_rho / best_alpha:.4f}")

        if best_project.cost + cost_selected_projects <= budget and best_project not in selected_projects:
            selected_projects.append(best_project)

        cost_selected_projects = sum(project.cost for project in selected_projects)

        for i, voter in enumerate(voters):
            u = get_utility(voter, best_project)
            if u > 0:
                virtual_budgets[i] = max(0, virtual_budgets[i] - best_rho * best_project.cost * u)
        logger.info(f"Updated virtual budgets: {[{str(v): round(b, 2)} for v, b in zip(voters, virtual_budgets)]}")

        budget_for_project = {
            project: sum(virtual_budgets[i] * get_utility(voter, project) for i, voter in enumerate(voters)) for
            project in all_projects}

        available_projects = [project for project in all_projects if
                              cost_selected_projects + project.cost <= budget and budget_for_project[
                                  project] > 0 and project not in selected_projects]
        logger.info(f"Selected projects: {selected_projects}\n")
    return selected_projects


def fractional_equal_shares(instance, profile):
    if not isinstance(profile, (ApprovalProfile, CardinalProfile)):
        raise TypeError("profile must be an instance of ApprovalProfile or CardinalProfile")
    if any(not isinstance(voter, (ApprovalBallot, CardinalBallot)) for voter in profile):
        raise TypeError("All items inside the profile must be ApprovalBallot or CardinalBallot instances")

    logger = logging.getLogger(__name__)
    logger.info("\nFractional equal shares")

    voters = sorted(list(profile), key=lambda v: str(v))
    cost_selected_projects = 0
    budget = instance.budget_limit
    num_voters = profile.num_ballots()
    virtual_budgets = [budget / num_voters for _ in voters]
    all_projects = sorted(list(instance), key=lambda p: str(p))
    logger.info(f"Budget: {budget}")
    logger.info(f"Virtual budgets: {[{str(v): round(b, 2)} for v, b in zip(voters, virtual_budgets)]}")

    budget_for_project = {
        project: sum(virtual_budgets[i] * get_utility(voter, project) for i, voter in enumerate(voters))
        for project in all_projects
    }

    project_part = {project: 0 for project in all_projects}

    available_projects = [
        project for project in all_projects
        if cost_selected_projects + project.cost * (1 - project_part[project]) <= budget and
           budget_for_project[project] > 0 and
           project_part[project] != 1
    ]

    while available_projects and cost_selected_projects < budget:
        logger.info(f"Remaining budget: {budget - cost_selected_projects}")
        project_utilities = {c: sum(get_utility(voter, c) * c.cost for voter in voters) for c in available_projects}
        valid_projects = [c for c in available_projects if project_utilities[c] > 0]

        if not valid_projects:
            break

        logger.info("Candidate projects evaluation details (rho = cost / utilities):")
        for project in valid_projects:
            current_rho = project.cost / project_utilities[project]
            logger.info(f"  -> Project {project.name}: rho = {current_rho:.4f}")

        c = min(valid_projects, key=lambda project: project.cost / project_utilities[project])
        p = c.cost / project_utilities[c]

        fractions = [1 - project_part[c]]
        for i, voter in enumerate(voters):
            denom = p * get_utility(voter, c) * c.cost
            if denom > 0:
                fractions.append(virtual_budgets[i] / denom)

        a = min(fractions)
        logger.info(f"Selected project: {c}")
        logger.info(f"alpha = {a:.4f}, rho = {p:.4f}")

        project_part[c] += a
        for i, voter in enumerate(voters):
            virtual_budgets[i] = max(0, virtual_budgets[i] - a * p * get_utility(voter, c) * c.cost)

        voters = [v for i, v in enumerate(voters) if virtual_budgets[i] > 0]
        virtual_budgets = [b for b in virtual_budgets if b > 0]
        logger.info(f"Updated virtual budgets: {[{str(v): round(b, 2)} for v, b in zip(voters, virtual_budgets)]}")
        budget_for_project = {
            project: sum(virtual_budgets[i] * get_utility(voter, project) for i, voter in enumerate(voters))
            for project in all_projects
        }
        cost_selected_projects = sum(project.cost * project_part[project] for project in all_projects)

        available_projects = [
            project for project in all_projects
            if cost_selected_projects + project.cost * (1 - project_part[project]) <= budget and
               budget_for_project[project] > 0 and
               project_part[project] != 1
        ]
        logger.info(f"Selected project parts: {project_part}\n")

    return dict(sorted(project_part.items(), key=lambda item: str(item[0])))


# --- NEW: VISUAL FAIRNESS REPORT GENERATION PIPELINES ---

def analyze_bos_ejr_up_to_t(instance, profile, result):
    """Generates structured structural analysis logs explaining EJR up to t properties."""
    if not instance or not list(profile):
        return []
    if not isinstance(list(profile)[0], ApprovalBallot):
        return None  # Signal that analysis is skipped for Cardinal Ballots

    reports = []
    c_max = max((p.cost for p in instance), default=0)
    n = profile.num_ballots()

    for idx, (group, project_set) in enumerate(cohesive_groups(instance, profile)):
        S_len = len(group)
        if S_len == 0: continue

        t = ((n - S_len) / (2 * S_len)) * c_max
        cost_T = sum(p.cost for p in project_set)
        T_minus_W = [p for p in project_set if p not in result]

        voters_info = []
        condition_met = False

        for voter in group:
            u_i_W = sum(p.cost for p in result if p in voter)
            voter_satisfied = False

            if not T_minus_W:
                if u_i_W >= cost_T - t:
                    voter_satisfied = True
                    condition_met = True
                desc = f"Utility {u_i_W} ≥ Cost(T)-t ({cost_T} - {t:.1f} = {cost_T - t:.1f})"
            else:
                voter_satisfied = all(u_i_W >= cost_T - t - c.cost for c in T_minus_W)
                if voter_satisfied:
                    condition_met = True
                desc = f"Utility {u_i_W} ≥ Cost(T)-t-Cost(c) for all missing projects c"

            voters_info.append({
                "ballot": ", ".join([p.name for p in voter]),
                "utility": u_i_W,
                "satisfied": voter_satisfied,
                "formula": desc
            })

        reports.append({
            "id": idx + 1,
            "size": S_len,
            "target_projects": ", ".join([p.name for p in project_set]),
            "cost_T": cost_T,
            "t_slack": round(t, 2),
            "missing_projects": ", ".join([p.name for p in T_minus_W]) if T_minus_W else "None (Fully Funded)",
            "is_fair": condition_met,
            "voters": voters_info
        })
    return reports


def analyze_fres_fractional_ejr(instance, profile, fres_result):
    """Generates structured structural analysis logs explaining Fractional EJR properties."""
    if not instance or not list(profile):
        return []
    if not isinstance(list(profile)[0], ApprovalBallot):
        return None

    reports = []
    # Handle if fres_result is a clean dict or wrap inside custom structures
    fractions = fres_result if isinstance(fres_result, dict) else {}

    for idx, (group, project_set) in enumerate(cohesive_groups(instance, profile)):
        S_len = len(group)
        if S_len == 0: continue

        cost_T = sum(p.cost for p in project_set)
        voters_info = []
        condition_met = False

        for voter in group:
            u_i_W_frac = sum(p.cost * fractions.get(p, 0.0) for p in instance if p in voter)
            voter_satisfied = u_i_W_frac >= cost_T
            if voter_satisfied:
                condition_met = True

            voters_info.append({
                "ballot": ", ".join([p.name for p in voter]),
                "utility": round(u_i_W_frac, 2),
                "satisfied": voter_satisfied,
                "formula": f"Fractional Utility {u_i_W_frac:.1f} ≥ Target Cost {cost_T}"
            })

        reports.append({
            "id": idx + 1,
            "size": S_len,
            "target_projects": ", ".join([p.name for p in project_set]),
            "cost_T": cost_T,
            "is_fair": condition_met,
            "voters": voters_info
        })
    return reports