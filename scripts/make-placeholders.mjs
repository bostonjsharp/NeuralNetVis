import { writeFileSync } from "node:fs";
// 1x1 deep-blue JPEG, base64. Placeholder art until real screenshots exist.
const jpg = Buffer.from(
  "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a" +
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA" +
    "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q==",
  "base64"
);
writeFileSync("wide.jpg", jpg);
writeFileSync("thumb.jpg", jpg);
console.log("wrote wide.jpg, thumb.jpg");
