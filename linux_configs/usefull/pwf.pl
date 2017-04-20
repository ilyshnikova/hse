#! /usr/bin/perl

use strict;
use warnings;

my $workdir = `pwd`;
unless (defined($ARGV[0])) {
	print "File name undefined\n";
	exit;
}

my $file = $ARGV[0];
chomp($workdir);

my $self_host_name = `hostname -f`;
chomp($self_host_name);
my $host = $ARGV[1] ? $self_host_name . ":" : '';

if (-e $workdir . "/" . $file) {
	$workdir =~ s/ /\\ /g;
	$file =~ s/ /\\ /g;
	print $host . $workdir . "/" . $file . "\n";
} else {
	print "No such file: $file\n";
}
