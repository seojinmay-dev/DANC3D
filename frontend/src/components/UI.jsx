export function Button({children,secondary=false,...props}) {
  return <button className={`button ${secondary?'secondary':''}`} {...props}>{children}</button>;
}

export function PageTitle({children,onBack}) {
  return <h1 className="page-title">
    {onBack&&<button className="back" onClick={onBack} aria-label="이전 화면">←</button>}
    {children}
  </h1>;
}
