#!/usr/bin/perl

use strict;
use warnings;

my $launch_string = join("", <>);
chomp($launch_string);

my ($env_array, $command) = get_env_and_rest_command($launch_string);
my ($user_string, $rest_command, $with_sudo) = get_user_string_and_rest_command($command);
launch_debugger($env_array, $user_string, $rest_command, $with_sudo);


sub get_home_directory {
	my $user_string = shift;

	my ($user) = ($user_string =~ /\w+\s+\-u\s+(\w+)/);
       	if (!defined($user) && $with_sudo) {
		$user = 'root';
	} elsif (!defined($user) && !$with_sudo) {
		$user = `whoami`;
	}
	chomp($user);

	my $homedir = `eval "echo ~$user"`;
	chomp($homedir);

	return $homedir;
}

sub launch_debugger {
	my $env_array = shift;
	my $user_string = shift;
	my $rest_command = shift;
	my $with_sudo = shift;

	my $user_home_dir = get_home_directory($user_string, $with_sudo);

	print "$user_string HOME=$user_home_dir $env_array rlwrap -t XTerms -H \"$user_home_dir/debug\" perl -d $rest_command\n";

#	exec("$user_string HOME=$user_home_dir PERL5LIB=. $env_array rlwrap -t XTerms -H \"$user_home_dir/debug\" perl -d $rest_command");
}

sub get_env_and_rest_command {
	my $launch_string = shift;

	my $env_array = [];

	while ($launch_string =~ m/(\S+\=\S*)/g) {
		my $found_string = $&;
		if (
			index($launch_string, $found_string) <= index($launch_string, '.pl') ||
			index($launch_string, $found_string) <= index($launch_string, '.cgi')
		) {
			push(@$env_array, $&);
		}
	}

	foreach my $env_var (@$env_array) {
		$launch_string =~ s/$env_var//g;
	}

	return (join(" ", @$env_array), $launch_string);
}

sub get_user_string_and_rest_command {
	my $command = shift;

	my $user_string = undef;
	my $rest_command = $command;

	my $with_sudo = ($command =~ /sudo/) ? 1 : 0;

	if ($command =~ /(sudo\s+(\-u\s+\w+){0,1})/) {
		$user_string = $1;
		$rest_command =~ s/$user_string//g;
	} else {
		$user_string = "";
	}

	return ($user_string, $rest_command, $with_sudo);
}

#user=$1
#if [ $2 ]; then
#	echo "Enter environment params"
#	read -e environment
#fi
#echo "Enter programm name to debug"
#read -e programm
#user_home_dir=`eval echo "~$user"`
#eval "HOME=$user_home_dir PERL5LIB=.  sudo -u $user $environment rlwrap -t XTerms -H "$user_home_dir/debug" perl -d $programm"
