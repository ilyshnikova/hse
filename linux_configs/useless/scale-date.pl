#!/usr/bin/perl

use strict;
use warnings;

my $first_date = shift;
my $second_date = shift;
my $third_date = shift;

my ($first_unixtime, $second_unixtime, $third_unixtime) = map {get_unixtime($_)} $first_date, $second_date, $third_date;

my $fourth_unixtime = $third_unixtime + ($second_unixtime - $first_unixtime);

print get_date($fourth_unixtime) . "\n";

sub get_unixtime {
	my $date = shift;

	my $unixtime = `date -d "$date" +%s`;
	chomp($unixtime);

	return $unixtime;
}

sub get_date {
	my $unixtime = shift;

	my $date = `date -d \@$unixtime`;
	chomp($date);

	return $date;
}


