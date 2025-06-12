import subprocess

def generate_mermaid_git_graph(simulated_git_log):
    # This is a simplified example. In a real scenario, you would run git log.
    # Here we simulate the output of git log --all --graph --pretty=format:%h,%d,%s
    # based on a simple history.
    mermaid_code = "gitGraph\n"
    commits_seen = {} # To track commits and avoid duplicates if needed

    for line in simulated_git_log.strip().split('\n'):
        line = line.strip()
        if line.startswith('*'):
            # Parse the commit line
            # Handle potential extra space after * and split by the first two commas
            parts = line[1:].strip().split(',', 2)
            if len(parts) >= 2:
                hash_val = parts[0].strip()
                refs = parts[1].strip()
                message = parts[2].strip() if len(parts) > 2 else ""

                commit_line = f'    commit id: "{hash_val}"'

                # Process references (branches, tags)
                if refs:
                    # Remove parentheses and split by comma
                    ref_list = [r.strip() for r in refs.replace('(', '').replace(')', '').split(',') if r.strip()]
                    processed_refs = []
                    for ref in ref_list:
                        if '->' in ref:
                            ref = ref.split('->')[-1].strip() # Get the branch name after ->
                        if ref and ref != 'HEAD': # Exclude the simple HEAD reference
                            processed_refs.append(f'"{ref}"')
                    if processed_refs:
                        # Join with comma and space as it's a single tag attribute
                        commit_line += f' tag: {", ".join(processed_refs)}'


                if message:
                     # Escape double quotes in message
                    message = message.replace('"', '\\"')
                    commit_line += f' msg: "{message}"'

                mermaid_code += commit_line + "\n"
                commits_seen[hash_val] = True

        # Note: Handling merge lines (|/ \) is more complex and not fully covered
        # in this simple parser, requires analyzing the graph structure.

    print(mermaid_code)

    
git log --all --graph --pretty=format:%h,%d,%s
simulated_git_log = """
* 4b0f846, (HEAD -> main, origin/main, origin/HEAD),优化了MerMaidChat
* e74d508,,update-version
* 80b7624,,优化了名称
* 440f090,,update-version
* 41186e7,,update-version
* 8edb42d,,up
* 14a22d4,,update
* 8092c3d,,update
* a1b6806,,Update prompts.py
* 15cb0d3,,update
* cd1745d,,update
* 608d378,,完整规整化V0.2
* 945d300,,备忘录V1
* 72c0262,,update
* cc855f8,,Update pyproject.toml
* 92edae8,,update
* 36e8763,,update version
* 3b4496a,,update
* 3864d6a,,update
* 2560f86,,init
* 33bdc9e,,init
* 38a4ca1,,Initial commit
* f604ac3, (origin/gh-pages, gh-pages),Deployed 945d300 with MkDocs version: 1.6.1
* 0670441,,Deployed 3b4496a with MkDocs version: 1.6.1
* 36fd11f,,Deployed 38a4ca1 with MkDocs version: 1.6.1
"""
generate_mermaid_git_graph(simulated_git_log)