import test from 'node:test';
import assert from 'node:assert/strict';
import {loginUser,registerUser} from '../src/services/api.js';

test('central auth helpers preserve the login form and registration query contracts',async t=>{
  const controller=new AbortController();
  t.mock.method(globalThis,'fetch',async(url,options)=>{
    assert.equal(options.method,'POST');assert.equal(options.signal,controller.signal);
    if(url==='/api/login') {
      assert.equal(options.headers['Content-Type'],'application/x-www-form-urlencoded');
      assert.equal(options.body.get('username'),'test user');assert.equal(options.body.get('password'),'test&password');
      assert.equal(options.body.get('grant_type'),'password');
      return {ok:true,json:async()=>({access_token:'token',nickname:'test user'})};
    }
    const parsed=new URL(url,'http://localhost');
    assert.equal(parsed.pathname,'/api/register');assert.equal(parsed.searchParams.get('username'),'test user');
    assert.equal(parsed.searchParams.get('password'),'test&password');
    return {ok:true,json:async()=>({message:'registered'})};
  });
  assert.equal((await loginUser('test user','test&password',{signal:controller.signal})).access_token,'token');
  assert.equal((await registerUser('test user','test&password',{signal:controller.signal})).message,'registered');
});

test('auth helpers report rejected credentials and duplicate registrations',async t=>{
  t.mock.method(globalThis,'fetch',async()=>({ok:false,status:400,json:async()=>({detail:'이미 존재하는 아이디입니다.'})}));
  await assert.rejects(loginUser('test','wrong'),/로그인 실패/);
  await assert.rejects(registerUser('test','password'),/이미 존재하는 아이디/);
});
