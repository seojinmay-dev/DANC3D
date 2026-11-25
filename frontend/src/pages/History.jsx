import {Button} from '../components/UI.jsx';
import PageLayout from '../components/PageLayout.jsx';
import RecordsTable from '../components/RecordsTable.jsx';
import useRecords from '../hooks/useRecords.js';

export default function History({guestId,navigate,identity}) {
  const history=useRecords(guestId,undefined,identity?.token);
  if(!identity)return <PageLayout title="연습 히스토리" description="로그인 후 이용할 수 있습니다."
    actions={<Button secondary onClick={()=>navigate('home')}>로그인 하러가기</Button>}>
    <p className="empty">로그인하면 계정에 저장된 연습 기록을 확인할 수 있습니다.</p>
  </PageLayout>;
  const actions=<>
    <Button secondary onClick={()=>navigate('songs')}>연습 시작</Button>
    <Button onClick={()=>navigate('home')}>홈으로</Button>
  </>;
  return <PageLayout title="연습 히스토리" description={`${identity.username}님의 저장된 연습 기록 · 최신순`}
    scrollLabel="전체 연습 기록" actions={actions}>
    {history.loading&&<p className="empty" role="status">연습 기록을 불러오는 중...</p>}
    {history.error&&<div className="camera-error" role="alert">{history.error}<Button secondary onClick={history.retry}>기록 다시 시도</Button></div>}
    {history.data&&<RecordsTable records={history.data}/>}
  </PageLayout>;
}
