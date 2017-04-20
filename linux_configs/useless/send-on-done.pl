#!/usr/bin/perl

use strict;
use warnings;

use POSIX qw(strftime);
use Getopt::Long;
use HTTP::Request::Common;
use LWP;
use URI::Escape;

use Yabs::Logger;
use Yabs::Funcs;

my $SEND_SMS_URL = "https://golem.yandex-team.ru/api/sms/send.sbml?resps=olegts";

check_installed_python_xmpp();

my $args = {};
GetOptions(
	$args,
	"pid=s",
	"name=s",
	"remote=s",
	"outer",
	"help",
);

if ($args->{'help'}) {
	print "
	pid=...
	name=...
	remote=...
	outer
";
	exit(0);
}

foreach my $key ("name", "pid") {
	exit ERROR "$key undefined" unless (defined($args->{$key}));
}

my $pid = $args->{'pid'};

if ($pid =~ /^\d+$/) {
    $pid = " $pid ";
}

INFO 1;
while(1) {
	INFO "Check done...";
	my $res = check_pid($pid)
	or ERROR "check_pid failed";

	if ($res > 0) {
		INFO "Done.";
		send_jabber($pid);
		exit(0);
	}
	INFO "Not done.";
	sleep(60);
}

sub check_pid {
	my $pid = shift;

	my $command = "ps aux | grep -E '$pid'  | grep -v $0 | grep -v grep | grep -v vim | grep -v less || true";

    INFO $command;

	if ($args->{'remote'}) {
		$command = "ssh " . $args->{'remote'} . " '$command'";
	}

	my $res = yabs_exec($command);

	unless($res->{'success'}) {
		return N_ERROR $res->{'errors'};
	}

	my $res_count = scalar(split(/\n/, $res->{'output'}));
	return ($res_count > 0) ? -1 : 1;
}

sub send_letter {
	my $pid = shift;
	SendEmail(
		"Script '" . $args->{'name'} . "' stopped at " . strftime("%Y-%m-%d %H:%M:%S", localtime(time())),
		"Processor pid $pid stopped to proceed",
		$args->{'outer'} ? "olegtsts\@gmail.com" : "olegts\@yandex-team.ru",
		'no',
		'text/html; charset=UTF-8'
	)
	or return N_ERROR "failed to send mail";

	return 1;
}

sub check_installed_python_xmpp {
	system("dpkg -l | grep python-xmpp")
	and die "python-xmpp is not installed";
}

sub send_jabber {
	my $pid = shift;

	if ($args->{'outer'}) {
		send_letter($pid)
		or return N_ERROR "send_letter failed";

		return 1;
	} else {
		my $host = GetHost();

		system("python ~/send_by_jabber.py olegts\@yandex-team.ru '$host:    Привет! Я сделал тебе задачу " . $args->{'name'} . "!'")
		and return N_ERROR "system failed";
	}

	return 1;
}

sub send_sms {
	my $text = shift;

	my $user_agent = LWP::UserAgent->new(ssl_opts => { verify_hostname => 0 });

	my $url = $SEND_SMS_URL . '&msg=' . uri_escape($text);
	INFO "Sending sms to olegts ($url)";

	my $content = request_post($user_agent, $url)
	or return N_ERROR "request_post $url failed";

	chomp($content);
	if ($content ne 'ok') {
		WARN "Send sms failed: content returned is $content";
	}

	return 1;
}

sub request_post {
	my $ua = shift;
	my $url = shift;

	my $uri = new URI($url);
	my $params = [$uri->query_form()];

	$url =~ s/\?.*//g;

	my $req = POST($url, $params);
	my $res = $ua->request($req);

	if ($res->is_success()) {
		return $res->content();
	} else {
		return N_ERROR "request_post failed";
	}
}
