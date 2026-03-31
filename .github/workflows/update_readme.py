#!/usr/bin/env python3

import json
import optparse

class UpdateReadme:
    def __init__(self, title, url, pr_num, created, last_updated, wip, num_commits, author):
        self.title = title
        self.url = url
        self.pr_num = pr_num
        self.created = created
        self.last_updated = last_updated
        self.wip = wip
        self.num_commits = num_commits
        self.author = author
        self.remote_repo_url = "https://github.com/roiser/madgraph4gpu-generated-processes/"

    def update_json(self):
        fh = open('readme_data.json', 'r')
        data = json.load(fh)
        fh.close()

        if self.pr_num not in data:
            new_entry = {
                "title": self.title,
                "branch": self.remote_repo_url + "tree/codegen-pr-" + str(self.pr_num),
                "url": self.url,
                "created": self.created,
                "last_updated": self.last_updated,
                "wip": self.wip,
                "num_commits": self.num_commits,
                "author": self.author
                }
            data[self.pr_num] = new_entry
        else:
            data[self.pr_num]['last_updated'] = self.last_updated
            data[self.pr_num]['wip'] = self.wip
            data[self.pr_num]['num_commits'] = self.num_commits
            data[self.pr_num]['title'] = self.title

        fh = open('readme_data.json', 'w')
        json.dump(data, fh, indent=4)
        fh.close()

    def write_readme(self):
        fh = open('readme_data.json', 'r')
        data = json.load(fh)
        fh.close()

        sorted_prs = sorted(data.items(), key=lambda x: x[1]['last_updated'], reverse=True)

        with open('README.md', 'w') as fh:
            fh.write("# Pull Request Summary\n\n")
            fh.write("| PR Number | Local Branch | Title | WIP | # Commits | Author | Created | Last Update |\n")
            fh.write("|-----------|--------------|-------|-----|-----------|--------|---------|-------------|\n")
            for pr_num, details in sorted_prs:
                fh.write(f"| [{pr_num}]({details['url']}) | [{details['branch'].split('/')[-1]}]({details['branch']}) | {details['title']} | {details['wip']} | {details['num_commits']} | {details['author']} | {details['created']} | {details['last_updated']} |\n")

    def run(self):
        self.update_json()
        self.write_readme()
        
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--title', dest='title', help='PR title')
    parser.add_option('--url', dest='url', help='PR URL')
    parser.add_option('--pr_num', dest='pr_num', help='PR number')
    parser.add_option('--created', dest='created', help='PR creation date')
    parser.add_option('--last_updated', dest='last_updated', help='PR last updated date')
    parser.add_option('--wip', dest='wip', help='Is PR WIP')
    parser.add_option('--num_commits', dest='num_commits', help='Number of commits in PR')
    parser.add_option('--author', dest='author', help='PR author') 

    options, _ = parser.parse_args()

    updater = UpdateReadme(options.title, options.url, options.pr_num, options.created, options.last_updated, options.wip, options.num_commits, options.author)
    updater.run()