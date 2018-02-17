{-# LANGUAGE RankNTypes #-}

applyToTuple :: (forall a. [a] -> b) -> ([p], [q]) -> (b, b)
applyToTuple f (a, b) =  (f a, f b)

main = do
  print $ applyToTuple length ("hello",[1,2,3])
