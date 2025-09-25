a) Basic Linux Commands
 Objective: To learn and execute fundamental Linux commands for file & directory 
 management.
1. pwd – Print Working Directory
❖ Explanation: Displays the full path of the current working directory.
❖ Syntax: pwd
❖ Code: pwd
2. ls – List Directory Contents
❖ Explanation: Lists files and directories.
❖ Syntax: ls [options] [directory]
❖ Code: ls -l, ls -a
3. cd – Change Directory
❖ Explanation: Navigates between directories.
❖ Syntax: cd [directory]
❖ Code: cd /home
4. mkdir – Make Directory
❖ Explanation: Creates a new directory.
❖ Syntax: mkdir [directory_name]
❖ Code: mkdir lab1
Cyber Security Lab Manual: Experiment – 1 2
Prof. Prasad Patil
Department of Computer Applications, Cyber Security Lab (24EBCE301), BCA 5
th sem
KLE Technological University, 
Belagavi.
5. rm – Remove File/Directory
❖ Explanation: Deletes files or directories.
❖ Syntax: rm [options] [file/directory]
❖ Code: rm file1.txt
 rm -r lab1
6. cp – Copy Files/Directories
❖ Explanation: Copies files or directories from one location to another.
❖ Syntax: cp [source] [destination]
❖ Code: cp file1.txt file2.txt
7. mv – Move/Rename Files
❖ Explanation: Moves or renames files and directories.
❖ Syntax: mv [source] [destination]
❖ Code: mv file2.txt Documents/
8. cat – Concatenate and Display
❖ Explanation: Displays contents of a file.
❖ Syntax: cat [filename]
❖ Code: cat /etc/os-release
9. chmod – Change Permissions
❖ Explanation: Modifies read, write, and execute permissions of 
files/directories.
❖ Syntax: chmod [permissions] [file]
❖ Code: chmod 755 file1.txt
10. history – Command History
❖ Explanation: Shows list of previously executed commands.
❖ Syntax: history
❖ Code: history
b) Networking Commands
 Objective : To configure, test, and analyze network connections.
1. ifconfig / ip – Display Network Info
❖ Syntax: ifconfig
 ip addr
2. ping – Test Connectivity
❖ Syntax: ping [hostname/IP]
❖ Code: ping google.com
3. netstat – Network Statistics
❖ Syntax:netstat [options]
❖ Code: netstat -tuln
Cyber Security Lab Manual: Experiment – 1 3
Prof. Prasad Patil
Department of Computer Applications, Cyber Security Lab (24EBCE301), BCA 5
th sem
KLE Technological University, 
Belagavi.
4. nmap – Network Scanner
❖ Syntax:nmap [options] [target]
❖ Code: nmap 192.168.1.1
5. traceroute – Trace Route
❖ Syntax: traceroute [hostname]
❖ Code: traceroute google.com
c) Package Management Commands
 Objective: To manage software packages using apt.
1. apt update – Updates package lists.
Code: sudo apt update
2. apt upgrade – Upgrades installed packages.
Code: sudo apt upgrade
3. apt install – Installs a package.
Code: sudo apt install nmap
4. apt remove – Removes a package (keeps config).
Code: sudo apt remove nmap
5. apt purge – Removes package & config.
Code: sudo apt purge nmap
d) File Permissions and Ownership Commands
 Objective: To demonstrate management of file permissions and ownership.
1. ls -l – Shows file permissions.
Code: ls -l
2. chmod – Change file permissions.
Code: chmod 744 file1.txt
3. chown – Change file ownership.
Code: sudo chown student:student file1.txt
e) Process Management Commands
 Objective: To manage running processes.
1. ps – Display processes.
Code: ps aux
2. top – Real-time process monitoring.
Code: top
Cyber Security Lab Manual: Experiment – 1 4
Prof. Prasad Patil
Department of Computer Applications, Cyber Security Lab (24EBCE301), BCA 5
th sem
KLE Technological University, 
Belagavi.
3. kill – Kill a process by PID.
Code: kill -9 1234
f) System Information Commands
 Objective: To retrieve system and user information.
1. uname – System info.
Code: uname -a
2. whoami – Current user.
Code: whoami
3. who – All logged-in users.
Code: who
4. uptime – System uptime.
Code: uptime
5. cal – Calendar.
Code: cal 2025
6. date – Current date/time.
Code: date
g) Text Editors Commands
 Objective: To create and edit files using command-line editors.
1. nano – Beginner-friendly editor.
Code: nano file1.txt
2. vi / vim – Advanced editor.
Code: vi file2.txt
 Press i → Insert mode
 Press Esc & type :wq → Save & exit