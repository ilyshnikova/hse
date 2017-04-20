#!/usr/bin/perl

use strict;
use warnings;
use autodie qw(open);

use Fcntl qw(:flock);

our $HOME = $ENV{HOME};
our $CONFIG_FILE = "$HOME/screen-config-file";
our $SSH_AUTH_SOCK = $ENV{SSH_AUTH_SOCK};
our $USER_NAME = `whoami`;
chomp($USER_NAME);

my $screen = shift;

mark_is_vim();
process_ssh_auth_sock($screen);
open_screen($screen);
system("/usr/bin/screen -RDS $screen")
and die "screen is wrongly deattached";
close_screen($screen);

sub lock_screen {
	open(my $fh, ">$HOME/screen_lock");

	flock($fh, LOCK_EX);

	return $fh;
}

sub unlock_screen {
	my $fh = shift;

	flock($fh, LOCK_UN);

	close($fh);
}

sub mark_is_vim {
	open(my $fh, ">$HOME/is_there_vim");
	print $fh ($ENV{MYVIMRC} ? 1 : 0) . "\n";
	close($fh);
}

sub process_ssh_auth_sock {
	my $screen = shift;

	unlink("$HOME/.ssh/ssh_auth_sock_$screen");
	system("ln -s $SSH_AUTH_SOCK $HOME/.ssh/ssh_auth_sock_$screen");
	system("pwd > $HOME/screen_pwd");
}

sub read_config {
	unless (-e $CONFIG_FILE) {
		system("touch $CONFIG_FILE");
	}

	open (my $fh, "<$CONFIG_FILE");

	my $config = {};
	while (my $line = <$fh>) {
		chomp($line);
		my ($screen, undef) = split(/\t/, $line);
		$config->{$screen} = 1;
	}
	close($fh);

	return $config;
}

sub write_config {
	my $config = shift;
	open(my $fh, ">$CONFIG_FILE");

	foreach my $screen (keys %$config) {
		print $fh "$screen\t$USER_NAME\n";
	}
	close($fh);
}

sub config_addon {
	my $screen = shift;
	my $addon = shift;

	my $lock = lock_screen();
	my $config = read_config();

	$config->{$screen} += $addon;

	if ($config->{$screen} <= 0) {
		delete($config->{$screen});
	}

	write_config($config);
	unlock_screen($lock);
}

sub open_screen {
	my $screen = shift;
	config_addon($screen, 1);
}

sub close_screen {
	my $screen = shift;
	config_addon($screen, -1);
}
