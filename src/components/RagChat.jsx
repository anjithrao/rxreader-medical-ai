import {useEffect, useRef, useState} from "react";
import {AnimatePresence, motion} from "framer-motion";
import {askMedicineChat} from "../api.js";

function formatSection(section) {
  if (!section) return "REFERENCE";

  return String(section).replace(/_/g, " ").toUpperCase();
}

/**
 * @param {{
 *   isOpen: boolean,
 *   medicine: string,
 *   onClose: () => void
 * }} props
 */
export default function RagChat({isOpen, medicine, onClose}) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;

    setQuestion("");
    setMessages([]);
    setSources([]);
    setError("");

    setTimeout(() => {
      inputRef.current?.focus();
    }, 100);
  }, [isOpen, medicine]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  async function handleSubmit(event) {
    event?.preventDefault();

    const cleanQuestion = question.trim();

    if (!cleanQuestion || isLoading) {
      return;
    }

    setError("");

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: cleanQuestion,
    };

    setMessages((previous) => [...previous, userMessage]);

    setQuestion("");
    setIsLoading(true);

    try {
      const data = await askMedicineChat(
        cleanQuestion,
        medicine ? [medicine] : [],
      );

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          role: "assistant",
          content:
            data.answer ||
            "The available reference notes did not provide an answer.",
        },
      ]);

      setSources(data.sources || []);
    } catch (chatError) {
      setError(
        chatError.message || "Unable to contact the medicine assistant.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleSuggestion(text) {
    setQuestion(text);

    setTimeout(() => {
      inputRef.current?.focus();
    }, 50);
  }

  if (!isOpen) {
    return null;
  }

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-[100] flex items-end justify-center bg-black/60 p-0 backdrop-blur-sm sm:items-center sm:p-4"
        initial={{opacity: 0}}
        animate={{opacity: 1}}
        exit={{opacity: 0}}
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) {
            onClose();
          }
        }}
      >
        <motion.section
          initial={{
            opacity: 0,
            y: 40,
            scale: 0.98,
          }}
          animate={{
            opacity: 1,
            y: 0,
            scale: 1,
          }}
          exit={{
            opacity: 0,
            y: 30,
            scale: 0.98,
          }}
          transition={{
            duration: 0.22,
          }}
          className="flex h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-t-3xl border border-white/10 bg-clinic-panel shadow-card sm:h-[720px] sm:rounded-3xl"
        >
          {/* Header */}
          <header className="flex items-center justify-between border-b border-white/10 bg-white/[0.04] px-5 py-4">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-clinic-cyan/40 bg-clinic-cyan/10 font-mono text-sm font-bold text-clinic-cyan">
                  AI
                </span>

                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-clinic-cyan">
                    Medicine Assistant
                  </p>

                  <h2 className="truncate font-mono text-lg font-bold text-clinic-bone">
                    {medicine || "Medicine Information"}
                  </h2>
                </div>
              </div>

              <p className="mt-2 text-xs text-clinic-muted">
                Answers are grounded in the available reference notes.
              </p>
            </div>

            <button
              type="button"
              onClick={onClose}
              aria-label="Close medicine assistant"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-white/10 text-clinic-muted transition hover:border-clinic-cyan/40 hover:text-clinic-cyan"
            >
              <svg
                viewBox="0 0 24 24"
                className="h-5 w-5"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M6 6l12 12M18 6L6 18"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6">
            {messages.length === 0 && (
              <div className="flex min-h-full flex-col items-center justify-center text-center">
                <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-clinic-cyan/30 bg-clinic-cyan/10 font-mono text-xl text-clinic-cyan shadow-glow">
                  ?
                </div>

                <h3 className="font-mono text-xl font-bold text-clinic-bone">
                  Ask about {medicine}
                </h3>

                <p className="mt-2 max-w-md text-sm leading-6 text-clinic-muted">
                  Ask about uses, side effects, precautions, or other
                  information covered by the available reference notes.
                </p>

                <div className="mt-6 flex max-w-lg flex-wrap justify-center gap-2">
                  {[
                    "What are the uses?",
                    "What are the side effects?",
                    "What precautions are listed?",
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => handleSuggestion(suggestion)}
                      className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-clinic-muted transition hover:border-clinic-cyan/40 hover:bg-clinic-cyan/10 hover:text-clinic-cyan"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-6 ${
                      message.role === "user"
                        ? "bg-clinic-cyan text-clinic-void"
                        : "border border-white/10 bg-white/[0.055] text-clinic-bone"
                    }`}
                  >
                    {message.role === "assistant" && (
                      <p className="mb-1 font-mono text-[9px] uppercase tracking-[0.2em] text-clinic-cyan">
                        RxReader AI
                      </p>
                    )}

                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl border border-white/10 bg-white/[0.055] px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-clinic-cyan">
                        Retrieving
                      </span>

                      <span className="flex gap-1">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-clinic-cyan" />
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-clinic-cyan [animation-delay:150ms]" />
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-clinic-cyan [animation-delay:300ms]" />
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div ref={messagesEndRef} />

            {/* Sources */}
            {sources.length > 0 && (
              <div className="mt-6">
                <div className="mb-3 flex items-center gap-2">
                  <div className="h-px flex-1 bg-white/10" />

                  <span className="font-mono text-[9px] uppercase tracking-[0.25em] text-clinic-cyan">
                    Retrieved Sources
                  </span>

                  <div className="h-px flex-1 bg-white/10" />
                </div>

                <div className="space-y-2">
                  {sources.map((source, index) => (
                    <details
                      key={`${source.medicine}-${source.section}-${index}`}
                      className="group rounded-xl border border-white/10 bg-white/[0.035]"
                    >
                      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3">
                        <div className="min-w-0">
                          <p className="font-mono text-xs font-bold uppercase text-clinic-bone">
                            {source.medicine}
                          </p>

                          <p className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.15em] text-clinic-cyan">
                            {formatSection(source.section)}
                          </p>
                        </div>

                        <span className="text-clinic-muted transition group-open:rotate-180">
                          ↓
                        </span>
                      </summary>

                      <div className="border-t border-white/10 px-4 py-3">
                        <p className="text-xs leading-5 text-clinic-muted">
                          {source.excerpt}
                        </p>
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div className="mt-4 rounded-xl border border-clinic-red/40 bg-clinic-red/10 p-3 text-xs leading-5 text-rose-100">
                {error}
              </div>
            )}
          </div>

          {/* Input */}
          <form
            onSubmit={handleSubmit}
            className="border-t border-white/10 bg-white/[0.025] p-3 sm:p-4"
          >
            <div className="flex items-end gap-2 rounded-2xl border border-white/10 bg-clinic-void/60 p-2 transition focus-within:border-clinic-cyan/40">
              <textarea
                ref={inputRef}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSubmit(event);
                  }
                }}
                rows={1}
                disabled={isLoading}
                placeholder={`Ask about ${medicine || "this medicine"}...`}
                className="max-h-28 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm text-clinic-bone outline-none placeholder:text-clinic-muted disabled:opacity-50"
              />

              <button
                type="submit"
                disabled={!question.trim() || isLoading}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-clinic-cyan text-clinic-void transition hover:shadow-glow disabled:cursor-not-allowed disabled:opacity-30"
                aria-label="Send question"
              >
                <svg
                  viewBox="0 0 24 24"
                  className="h-4 w-4"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M4 12h15M13 6l6 6-6 6"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>

            <p className="mt-2 px-2 text-center font-mono text-[9px] uppercase tracking-[0.12em] text-clinic-muted">
              Reference-grounded information • Not a diagnosis
            </p>
          </form>
        </motion.section>
      </motion.div>
    </AnimatePresence>
  );
}
