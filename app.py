import io
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from pabutools.election import Project, Instance, ApprovalProfile, ApprovalBallot, CardinalProfile, CardinalBallot

# Import your algorithm implementations
from bos_equal_shares import bos_equal_shares, fractional_equal_shares

app = Flask(__name__)
app.secret_key = 'super_secret_dev_key'  # Required for flash messages


def parse_inputs(projects_str, ballots_str, budget_str, ballot_type):
    """Helper to convert raw form strings into correct pabutools objects."""
    try:
        budget = float(budget_str.strip())
    except ValueError:
        raise ValueError("Budget must be a valid numeric value.")

    # Parse projects: format "name:cost, name:cost"
    projects_dict = {}
    try:
        for item in projects_str.split(','):
            if not item.strip(): continue
            name, cost = item.split(':')
            p = Project(name.strip(), float(cost.strip()))
            projects_dict[name.strip()] = p
    except Exception:
        raise ValueError("Projects format must match 'p1:1000, p2:500'.")

    if not projects_dict:
        raise ValueError("You must provide at least one valid project.")

    ballots = []
    try:
        for line in ballots_str.strip().split('\n'):
            if not line.strip(): continue

            if ballot_type == 'cardinal':
                # Parse Cardinal format: "p1:100, p2:2"
                voter_scores = {}
                for pair in line.split(','):
                    if not pair.strip(): continue
                    p_name, score_str = pair.split(':')
                    p_name = p_name.strip()
                    if p_name in projects_dict:
                        voter_scores[projects_dict[p_name]] = float(score_str.strip())
                    else:
                        raise ValueError(
                            f"Project '{p_name}' specified in ballots does not exist in the projects list.")
                if voter_scores:
                    ballots.append(CardinalBallot(voter_scores))
            else:
                # Parse Approval format: "p1,p2"
                approved_names = [name.strip() for name in line.split(',') if name.strip()]
                approved_projects = {projects_dict[name] for name in approved_names if name in projects_dict}
                if len(approved_names) != len(approved_projects):
                    missing = [n for n in approved_names if n not in projects_dict]
                    raise ValueError(f"Projects {missing} specified in ballots do not exist in the projects list.")
                if approved_projects:
                    ballots.append(ApprovalBallot(approved_projects))
    except Exception as e:
        raise ValueError(f"Error parsing voter profile rows. Details: {e}")

    if not ballots:
        raise ValueError("You must provide at least one valid voter ballot line.")

    instance = Instance(list(projects_dict.values()), budget)
    profile = CardinalProfile(ballots) if ballot_type == 'cardinal' else ApprovalProfile(ballots)
    return instance, profile


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run_algorithm():
    algo_type = request.form.get('algo_type')
    ballot_type = request.form.get('ballot_type')  # 'approval' or 'cardinal'
    projects_raw = request.form.get('projects')
    ballots_raw = request.form.get('ballots')
    budget_raw = request.form.get('budget')

    # Intercept system out logging streams in real time
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(message)s'))

    target_logger = logging.getLogger('bos_equal_shares')
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(handler)

    try:
        # 1. Parse string inputs safely using our updated schema converter
        instance, profile = parse_inputs(projects_raw, ballots_raw, budget_raw, ballot_type)

        # 2. Execute selected logic path
        if algo_type == 'BOS':
            result = bos_equal_shares(instance, profile)
            result_display = ", ".join(
                [p.name for p in result]) if result else "No projects selected within budget constraints."
        elif algo_type == 'FRES':
            result = fractional_equal_shares(instance, profile)
            result_display = ", ".join([f"{p.name}: {round(fraction, 2)}" for p, fraction in result.items()])
        else:
            raise ValueError("Invalid core algorithm engine selection token.")

        # Clean up handlers and isolate output buffer metrics
        target_logger.removeHandler(handler)
        captured_logs = log_stream.getvalue()

        return render_template('result.html',
                               algo=algo_type,
                               ballot_mode=ballot_type.upper(),
                               budget=budget_raw,
                               projects=projects_raw,
                               ballots=ballots_raw,
                               result=result_display,
                               logs=captured_logs)

    except Exception as e:
        target_logger.removeHandler(handler)
        flash(f"Input Processing Error: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=6869)