# ~/.profile: executed by the command interpreter for login shells.
# This file is not read by bash(1), if ~/.bash_profile or ~/.bash_login
# exists.
# see /usr/share/doc/bash/examples/startup-files for examples.
# the files are located in the bash-doc package.

# the default umask is set in /etc/profile; for setting the umask
# for ssh logins, install and configure the libpam-umask package.
#umask 022

# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
	. "$HOME/.bashrc"
    fi
fi

bind -r '\C-s'
stty -ixon

for package in python-xmpp libnet-address-ip-local-perl python-dnspython
do
	if [ `dpkg -s $package 2>/dev/null | wc -l` == "0" ]; then echo -e "\n\t*** run sudo apt-get install $package ***\n"; fi
done


function svngrep() {
	clear && echo  && grep -E "$@" -r * | grep -v svn | grep -v debian | sed 's/:.*//g' | uniq
}

alias vimstore="perl $HOME/vimstore.pl"

#if which git >/dev/null; then
#	git config --global diff.tool vimdiff
#	git config --global difftool.prompt false
#	git config --global alias.svndiff difftool
#	alias gitdiff="git svndiff --cached"
#	git config --global core.editor "vim"
#fi

export SVN_EDITOR=vim

wdiff () {
	exec wdiff -n  -w $'\033[30;31m' -x $'\033[0m' -y $'\033[30;32m' -z $'\033[0m' $@ |less -R
}

alias install="DEBIAN_FRONTEND=noninteractive sudo apt-get install --assume-yes --force-yes -q0 -y --reinstall ";
alias send="${HOME}/send"
alias perld="${HOME}/perld"
alias bell="${HOME}/bell"
alias mysql=mysql\ -uroot
alias pwf=perl\ ~/pwf.pl
alias rotate=perl\ ~/rotate.pl
alias bscoll-ssh=~/bscoll-ssh
alias debuild="debuild --check-dirname-level 0 --no-lintian -i -us -uc -b"
alias ccp=perl\ ~/ccp.pl
alias ccp-get=perl\ ~/ccp-get.pl
alias rotate=perl\ ~/rotate.pl
alias syn-check=perl\ ~/syn-check.pl
alias pbcopy=perl\ ~/pbcopy.pl
source ~/.bash_aliases


alias screen="perl $HOME/screen.pl"

function screen2() {
        if [ $MYVIMRC ] ; then
                echo '1' > ~/is_there_vim;
        else
                echo '0' > ~/is_there_vim;
        fi
        rm -rf $HOME/.ssh/ssh_auth_sock_$@
        ln -s "$SSH_AUTH_SOCK" "$HOME/.ssh/ssh_auth_sock_$@"
        pwd > ~/screen_pwd
        /usr/bin/screen -RDS "$@"
}


#alias screen='if [ $MYVIMRC ] ; then echo '1' > ~/is_there_vim; else echo '0' > ~/is_there_vim; fi && /usr/bin/screen -RDS'

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

if [ -d "$HOME/arc/trunk/arcadia/devtools/ya" ] ; then
    PATH="$HOME/arc/trunk/arcadia/devtools/ya:$PATH"
fi

if [ -d "$HOME/arc/trunk/arcadia/devtools/codesearch/scripts" ] ; then
    PATH="$HOME/arc/trunk/arcadia/devtools/codesearch/scripts:$PATH"
fi

if [ -d "$HOME/miniconda2/bin" ]; then
    PATH="$HOME/miniconda2/bin:$PATH"
fi

#export EDITOR=vim
#ranger() {
#             command ranger $@ &&
#                        cd "$(grep \^\' ~/.config/ranger/bookmarks | cut -b3-)"
#}

alias svndiff="svn --diff-cmd=$HOME/svnvimdiff diff"

if [ $STY ] ; then
        stripped_ps=`echo $PS1 | cut -d'$' -f 1`
	additional_message="$STY"
        screen_name=`echo $STY | sed 's/^[0-9.]\+//g'`
        echo "reexport SSH_AUTH_SOCK"
        export SSH_AUTH_SOCK="$HOME/.ssh/ssh_auth_sock_$screen_name"
	if [ `cat ~/is_there_vim` -eq "1" ]; then
		additional_message="$STY :: VIM"
	fi
        PS1="$stripped_ps${debian_chroot:+($debian_chroot)}\u@\h [ $additional_message ] :\w$ "
else
	eval $LC_PASSED_COMMAND
fi



complete -C "perl -e '@w=split(/ /,\$ENV{COMP_LINE},-1);\$w=pop(@w);for(qx(screen -ls)){print qq/\$1\n/ if (/^\s*\d+\.\$w/&&/\d+\.([\w\-\.]+)/)}'" screen
export GIT_SSL_NO_VERIFY=1

ulimit -c unlimited
