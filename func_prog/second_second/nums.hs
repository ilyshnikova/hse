import Data.List

first = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]

second = ["", "десять", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]

third = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

fourth = ["тысяч", " тысяча"] ++ take 3 (repeat "тысячи") ++ take 5 (repeat "тысяч")


saynumber_small num =
	[third !! (num `div` 100)] ++
	(
		if (num `mod` 100) <= 19 then
			[first !! (num `mod` 100)]
		else
			[second !! ((num `div` 10) `mod` 10), first !! (num `mod` 10)]
	)

saynumber_list num =
	(
		if (num >= 1000) then
			saynumber_small (num `div` 1000) ++ [fourth !! ((num `div` 1000) `mod` 10)]
		else
			[]
	)
	++ saynumber_small (num `mod` 1000)

saynumber 0 = "ноль"
saynumber num = concat (intersperse " " (filter (\x -> x /= "") (saynumber_list num)))

main = do
  putStrLn $ saynumber 123456
