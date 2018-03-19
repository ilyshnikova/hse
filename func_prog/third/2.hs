data RegExp = Chr Char
            | Con RegExp RegExp
            | Uni RegExp RegExp
            | Eps
            | Ite RegExp

type NFA = [[(Maybe Int, Maybe Char)]]

replace_nothing x = map (map f) where
  f (Nothing, c) = (Just x, c)
  f p = p

inc_indexes inc = map (map f) where
  f (Just x, c) = (Just (x + inc), c)
  f p = p


build :: RegExp -> NFA
build Eps = [[(Nothing, Nothing)]]

build (Con a b) = replace_nothing (length a_nfa) a_nfa ++ inc_indexes (length a_nfa) b_nfa  where
  a_nfa = build a
  b_nfa = build b

build (Uni a b) = ((Just (length xs + 1), Nothing) : x) : xs ++ inc_indexes (length xs + 1) b_nfa  where
  x:xs = build a
  b_nfa = build b

build (Ite a) = ((Nothing, Nothing) : x) : xs   where
  x:xs = replace_nothing 0 (build a)

build (Chr a) = [[(Nothing, Just a)]]


eps_tr :: NFA -> [Maybe Int] -> [Maybe Int]
eps_tr nfa state = foldl step [] state where
  step xs Nothing | elem Nothing xs = xs
                  | otherwise = Nothing : xs

  step xs (Just i) | elem (Just i) xs = xs
                   | otherwise = foldl step ((Just i) : xs) $ map fst (filter (\(x, y) -> y == Nothing) (nfa !! i))


one_step :: NFA -> Char -> Maybe Int -> [Maybe Int]
one_step nfa c Nothing = []
one_step nfa c (Just i) = map fst (filter (\(x, y) -> y == Just c) (nfa !! i))

step :: NFA -> [Maybe Int] -> Char -> [Maybe Int]
step nfa states c = eps_tr nfa (concat (map (one_step nfa c) states))

match :: NFA -> String -> Bool
match nfa s = elem Nothing (foldl (step nfa) (eps_tr nfa [Just 0]) s)

match_re :: RegExp -> String -> Bool
match_re = match . build

main = do
  let re = Uni (Con (Ite (Chr 'a')) (Ite (Chr 'b'))) (Ite (Chr 'c'))
  print (match_re re "aba")
