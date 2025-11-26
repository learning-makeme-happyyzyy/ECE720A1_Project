import sys
import csv
from typing import Dict, Callable, List

import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
import statistics as st
import re

csv.field_size_limit(2**31 - 1)


def extract_rminer_data() -> Dict:
    commits_to_refactors = {}
    with open('rMinerRefactorings.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
        for row in reader:
            commit_id = row['commit_id']
            if commit_id in commits_to_refactors:
                commits_to_refactors[commit_id].append(row)
            else:
                commits_to_refactors[commit_id] = [row]
    return commits_to_refactors


def extract_r_commits() -> Dict:
    repos_to_commits = {}
    with open('refactoring_commits.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
        for row in reader:
            repo_url = row['url']
            commit = row['commit_id']
            if repo_url in repos_to_commits:
                repos_to_commits[repo_url].append(commit)
            else:
                repos_to_commits[repo_url] = [commit]
    return repos_to_commits


def get_repos_to_commits_to_refactors(refactor_filter_func: Callable[[Dict], List | str]) -> Dict:
    commits_to_refactors = extract_rminer_data()
    repos_to_commits = extract_r_commits()
    repos_to_commits_to_refactors = {}
    for commit, refactor_list in commits_to_refactors.items():
        for repo, commit_list in repos_to_commits.items():
            if commit in commit_list:
                if repo not in repos_to_commits_to_refactors:
                    repos_to_commits_to_refactors[repo] = {}
                repos_to_commits_to_refactors[repo][commit] = refactor_filter_func(refactor_list)
    return repos_to_commits_to_refactors


def convert_to_repos_to_refactoring_types(repos_to_commits_to_refactors: Dict):
    repo_to_refactoring_type = {}
    types = []
    for repo, commits in repos_to_commits_to_refactors.items():
        if repo not in repo_to_refactoring_type:
            repo_to_refactoring_type[repo] = {}
        for commit, refactoring_list in commits.items():
            for refactor in refactoring_list:
                r_type = refactor['type']
                if r_type not in types:
                    types.append(r_type)
                if r_type not in repo_to_refactoring_type[repo]:
                    repo_to_refactoring_type[repo][r_type] = 0
                repo_to_refactoring_type[repo][r_type] += 1
    return repo_to_refactoring_type, types


def create_plot_test_types():
    # filter refactorings
    r = get_repos_to_commits_to_refactors(lambda refactor_list: list(filter(lambda refactor: refactor['test'] == 'True',
                                                                            refactor_list)))
    repos_to_refactorings, refactoring_types = convert_to_repos_to_refactoring_types(r)
    data = []
    for ref_type in refactoring_types:
        boxplot = []
        for refactoring_dict in repos_to_refactorings.values():
            if ref_type in refactoring_dict:
                boxplot.append(refactoring_dict[ref_type])
            else:
                boxplot.append(0)
        data.append(boxplot)

    fig = plt.figure(figsize=(20, 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, 86), refactoring_types, rotation=90)
    plt.title("Testing Refactoring Types")
    plt.savefig("test_types_no_outliers.png", bbox_inches="tight")
    # print(repos_to_refactorings)


def create_plot_test_types_co_occurrence(get_top_most):
    num_top = 10
    def check_co_occurence(ref_list):
        has_production = False
        has_test = False
        for refactor in ref_list:
            if refactor['test'] == 'True':
                has_test = True
            else:
                has_production = True
        # has both production and test refactorings for co-occurrence
        return has_production and has_test

    # filter refactorings
    r = get_repos_to_commits_to_refactors(
        lambda refactor_list: list(filter(lambda refactor: refactor['test'] == 'True',
                                          refactor_list)) if check_co_occurence(refactor_list) else [])
    repos_to_refactorings, refactoring_types = convert_to_repos_to_refactoring_types(r)
    data = []
    for ref_type in refactoring_types:
        boxplot = []
        for refactoring_dict in repos_to_refactorings.values():
            if ref_type in refactoring_dict:
                boxplot.append(refactoring_dict[ref_type])
            else:
                boxplot.append(0)
        data.append(boxplot)
    print(len(refactoring_types))

    if get_top_most:
        ref_type_info = []
        for index, box in enumerate(data):
            ref_type_info.append({"name": refactoring_types[index], "amount": sum(box), "box": box,
                                  "median": st.median(box)})
        ref_type_info = sorted(ref_type_info, key=lambda x: x["median"], reverse=True)[:num_top]
        data = [ref["box"] for ref in ref_type_info]
        refactoring_types = [ref["name"] for ref in ref_type_info]

    fig = plt.figure(figsize=(6, 4))
    boxprops = dict(color="black", linewidth=1, facecolor="#A0AECA")
    medianprops = dict(color="black", linewidth=1)
    plt.boxplot(data, showfliers=True, vert=False, boxprops=boxprops, medianprops=medianprops, patch_artist=True)
    plt.yticks(range(1, num_top + 1 if get_top_most else len(refactoring_types) + 1),
               [re.sub(r'(\w+ \w+ )', '\\1\n', r_type) for r_type in refactoring_types])
    plt.gca().invert_yaxis()
    plt.xlim([0, 1100])
    plt.xlabel("Frequency")
    plt.savefig("co-occurring_test_types_top.pdf" if get_top_most else "co-occuring_test_types.pdf",
                bbox_inches="tight")


def get_missing_projects(the_projects: List):
    myclient = MongoClient('mongodb://10.0.0.101:27017/')
    mydb = myclient.smartshark_2_1
    all_projects_cursor = mydb.project.find()

    project_names = [a_project["name"] for a_project in all_projects_cursor]
    all_projects = the_projects.copy()

    for name in project_names:
        filter_projs = the_projects.copy()
        print(filter_projs)
        projects_w_name = filter(lambda x: x["name"] == name, filter_projs)
        if len([project for project in projects_w_name]) < 1:
            missing_project = mydb.project.find_one({"name": name})
            vcs_missing_project = mydb.vcs_system.find_one({"project_id": missing_project['_id']})
            print(vcs_missing_project)
            num_commits_missing_proj = mydb.commit.count_documents({"vcs_system_id": vcs_missing_project['_id']})
            missing_proj = {"total_commits": num_commits_missing_proj, "total_ref": 0}
            all_projects.append(missing_proj)
        else:
            all_projects.extend(projects_w_name)

    print(all_projects)

    commit_num_list = [project["total_commits"] for project in all_projects]
    print(f"Min: {np.min(commit_num_list)}, Q1: {np.percentile(commit_num_list, 25, method='midpoint')}"
          f"Median: {np.median(commit_num_list)}, Q3: {np.percentile(commit_num_list, 75, method='midpoint')}"
          f"Max: {np.max(commit_num_list)}")
    print(len(commit_num_list))

    ref_num_list = [project["total_ref"] for project in all_projects]
    print(f"Min: {np.min(ref_num_list)}, Q1: {np.percentile(ref_num_list, 25, method='midpoint')}"
          f"Median: {np.median(ref_num_list)}, Q3: {np.percentile(ref_num_list, 75, method='midpoint')}"
          f"Max: {np.max(ref_num_list)}")
    print(len(ref_num_list))

def classify_commits_and_plot():
    def handle_ref(ref_list):
        productions = 0
        tests = 0
        for ref in ref_list:
            if ref['test'] == "True":
                tests += 1
            elif ref['test'] == "False":
                productions += 1
            else:
                raise Exception("gello")
        if productions > 0 and tests > 0:
            return "co-occur"
        if productions > 0:
            return "production-only"
        if tests > 0:
            return "test-only"
    r = get_repos_to_commits_to_refactors(lambda refactor_list: handle_ref(refactor_list))

    stats = {}
    for repo, commit_list in r.items():
        stats[repo] = {"co-occur": 0, "production": 0, "test": 0, "total_ref": 0}
        for commit, r_type in commit_list.items():
            if r_type == "co-occur":
                stats[repo]["co-occur"] += 1
            if r_type == "production-only":
                stats[repo]["production"] += 1
            if r_type == "test-only":
                stats[repo]["test"] += 1
            stats[repo]["total_ref"] += 1

    client = MongoClient('mongodb://10.0.0.101:27017/')
    db = client.smartshark_2_1

    projects = []
    for repo_url, statistics in stats.items():
        vcs = db.vcs_system.find_one({'url': repo_url})
        proj_name = db.project.find_one({'_id': vcs['project_id']})['name']

        num_commits = db.commit.count_documents({"vcs_system_id": vcs["_id"]})
        projects.append({"name": proj_name, "total_commits": num_commits,
                         "test": str(round((statistics["test"]/statistics["total_ref"]) * 100, 2)),
                         "production": str(round((statistics["production"]/statistics["total_ref"]) * 100, 2)),
                         "co-occur": str(round((statistics["co-occur"]/statistics["total_ref"]) * 100, 2)),
                         "total_ref": statistics["total_ref"]})

    print(sorted(projects, key=lambda d: d["total_ref"], reverse=True))

    get_missing_projects(projects)  # print out # commits + # ref commits with missing projects that have no ref commits
    boxplot_data = [[float(project['production']) for project in projects],
                    [float(project['test']) for project in projects],
                    [float(project['co-occur']) for project in projects]]
    fig = plt.figure(figsize=(5, 2))
    boxprops = dict(color="black", linewidth=1, facecolor="#A0AECA")
    medianprops = dict(color="black", linewidth=1)
    plt.boxplot(boxplot_data, vert=False, boxprops=boxprops, medianprops=medianprops, patch_artist=True, widths=0.6)
    plt.yticks(range(1, 4), ['Source', 'Test',
                             'Co-Occurring'])
    plt.gca().invert_yaxis()
    plt.xlabel("Percentage of Commits")
    plt.ylabel("Refactoring Commits")
    plt.savefig("PreOfCommitsType.pdf", bbox_inches="tight")

    with open('project_stats.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['name', 'total_commits', 'total_ref', 'test', 'production',
                                                     'co-occur'])
        writer.writeheader()
        for value in sorted(projects, key=lambda d: d["total_ref"], reverse=True):
            writer.writerow(value)

if __name__ == '__main__':
    # step 1
    #classify_commits_and_plot()

    # step 2
    create_plot_test_types_co_occurrence(True)

