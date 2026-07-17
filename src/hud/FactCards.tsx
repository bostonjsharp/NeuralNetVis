import { useEffect, useState } from "react";

const FACTS = [
  "This network really is doing the math right now — thousands of multiplications for every guess.",
  "Every line is a weight the network learned by studying 60,000 handwritten digits.",
  "Orange connections excite the next neuron. Blue ones hold it back.",
  "Your drawing becomes just 784 numbers — one brightness per pixel. That's all it sees.",
  "Each middle neuron learned to spot its own little pattern — a curve here, a stroke there.",
  "The final step, softmax, turns raw scores into the percentages you see.",
  "No one told it what a 7 looks like. It figured that out from examples alone.",
  "Big AI models work the same way — just with billions of connections instead of thousands.",
  "When it's wrong, it's wrong confidently sometimes. Even small brains hallucinate.",
  "Brighter neurons are more activated — you're watching the network think.",
  "This wall holds four different brains — watch it swap between them and compare their scores.",
  "The straight-through brain has no hidden layers. It can only match overall shapes — and it shows.",
  "More neurons usually means smarter — but every extra layer is a chance to build up bigger ideas.",
  "Depth beats width for understanding: two small layers outscore one big vote almost every time.",
];

const ROTATE_MS = 12_000;

/** Rotating two-sentence education, one card at a time, bottom of the wall. */
export default function FactCards() {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => setIndex((i) => (i + 1) % FACTS.length), ROTATE_MS);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="fact-card" key={index}>
      {FACTS[index]}
    </div>
  );
}
