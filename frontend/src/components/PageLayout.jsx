import {PageTitle} from './UI.jsx';

export default function PageLayout({title,description,onBack,toolbar,actions,children,
  className='',scrollLabel='페이지 내용',busy}) {
  return <section className={`page-layout ${className}`} aria-busy={busy}>
    {title&&<header className="page-heading">
      <PageTitle onBack={onBack}>{title}</PageTitle>
      {description&&<p className="subtitle">{description}</p>}
    </header>}
    {toolbar&&<div className="page-toolbar">{toolbar}</div>}
    <div className="page-scroll" role="region" aria-label={scrollLabel} tabIndex={0}>{children}</div>
    {actions&&<div className="page-actions">{actions}</div>}
  </section>;
}
